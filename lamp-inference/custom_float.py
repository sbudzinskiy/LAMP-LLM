import torch
import triton
import triton.language as tl

def round_mantissa_(x: torch.Tensor, m_bits: int) -> torch.Tensor:
    TOTAL_MANTISSA_BITS = 23
    k = TOTAL_MANTISSA_BITS - m_bits
    
    if k <= 0:
        return x

    u = x.view(torch.int32)
    
    sign_mask = -2147483648 # 0x80000000
    abs_mask = 2147483647   # 0x7FFFFFFF
    exponent_mask = 0xFF << 23

    signs = u & sign_mask
    is_nan_inf = (u & exponent_mask) == exponent_mask
    nan_inf_vals = u[is_nan_inf].clone() 

    mask_k = (1 << k) - 1
    threshold = 1 << (k - 1)

    u.bitwise_and_(abs_mask)

    # RNE Rounding
    remainder = u & mask_k
    last_kept_bit = (u >> k) & 1
    round_up = (remainder > threshold) | ((remainder == threshold) & (last_kept_bit == 1))    
    u.add_(round_up.int() << k)
    u.bitwise_and_(~mask_k)
    u.bitwise_or_(signs)
    u[is_nan_inf] = nan_inf_vals

    return x

#############################################################################################################################

@triton.jit
def round_mantissa_triton(x, m_bits: tl.constexpr):
    TOTAL_MANTISSA_BITS = 23
    k = TOTAL_MANTISSA_BITS - m_bits    

    u = x.to(tl.int32, bitcast=True)

    sign_mask = -2147483648 # 0x80000000
    abs_mask = 2147483647   # 0x7FFFFFFF
    exponent_mask = 0xFF << 23
    
    s = u & sign_mask
    u_abs = u & abs_mask    
    is_nan_inf = (u_abs & exponent_mask) == exponent_mask
    
    mask_k = (1 << k) - 1
    threshold = 1 << (k - 1)
    
    remainder = u_abs & mask_k
    last_kept_bit = (u_abs >> k) & 1
    
    round_up = (remainder > threshold) | ((remainder == threshold) & (last_kept_bit == 1))    
    u_abs_rounded = u_abs + (round_up.to(tl.int32) << k)
    u_abs_final = u_abs_rounded & (~mask_k)
    
    u_abs_safe = tl.where(is_nan_inf, u_abs, u_abs_final)
    
    out_int = s | u_abs_safe
    return out_int.to(tl.float32, bitcast=True)

@triton.jit
def custom_accum_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_a_batch, stride_b_batch, stride_c_batch,
    BLOCK_SIZE_M: tl.constexpr, 
    BLOCK_SIZE_N: tl.constexpr, 
    BLOCK_SIZE_K: tl.constexpr,
    m_bits: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1)
    
    a_ptr_base = a_ptr + batch_id * stride_a_batch
    b_ptr_base = b_ptr + batch_id * stride_b_batch
    c_ptr_base = c_ptr + batch_id * stride_c_batch

    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    current_a_ptr = a_ptr_base + (offs_m * stride_am) 
    current_b_ptr = b_ptr_base + (offs_n * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        remaining_k = K - k * BLOCK_SIZE_K
        
        if BLOCK_SIZE_K >= 16:
            a_ptrs_block = current_a_ptr[:, None] + (offs_k[None, :] * stride_ak)
            b_ptrs_block = (offs_k[:, None] * stride_bk) + current_b_ptr[None, :]
            
            k_active = offs_k < remaining_k
            mask_a = mask_m[:, None] & k_active[None, :]
            mask_b = k_active[:, None] & mask_n[None, :]

            a = tl.load(a_ptrs_block, mask=mask_a, other=0.0)
            b = tl.load(b_ptrs_block, mask=mask_b, other=0.0)
            
            accumulator += tl.dot(a, b)
            accumulator = round_mantissa_triton(accumulator, m_bits)
        else:
            for i in range(BLOCK_SIZE_K):
                is_valid_k = i < remaining_k
                ptr_a_col = current_a_ptr + (i * stride_ak)                 
                ptr_b_row = current_b_ptr + (i * stride_bk)
                
                vec_a = tl.load(ptr_a_col, mask=(mask_m & is_valid_k), other=0.0)
                vec_b = tl.load(ptr_b_row, mask=(mask_n & is_valid_k), other=0.0)
                
                accumulator += vec_a[:, None] * vec_b[None, :]
                accumulator = round_mantissa_triton(accumulator, m_bits)
        
        current_a_ptr += BLOCK_SIZE_K * stride_ak
        current_b_ptr += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr_base + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, accumulator, mask=c_mask)

def custom_accum_gemm(a: torch.Tensor, b: torch.Tensor, m_bits: int, block_size_k: int = 32):
    """
    Computes C = A @ B supports multi-dimensional batching.
    e.g., (B, H, M, K) @ (B, H, K, N) -> (B, H, M, N)
    """
    assert a.shape[-1] == b.shape[-2], f"K dim mismatch: {a.shape[-1]} vs {b.shape[-2]}"
    assert a.is_cuda and b.is_cuda, "Inputs must be on CUDA"

    if b.ndim == 2:
        stride_b_batch = 0
        b_in = b.contiguous()
    else:
        assert a.shape[:-2] == b.shape[:-2], f"Batch dims mismatch: {a.shape[:-2]} vs {b.shape[:-2]}"
        stride_b_batch = -1

    batch_dims = a.shape[:-2] 
    M, K = a.shape[-2], a.shape[-1]
    N = b.shape[-1]

    a_in = a.reshape(-1, M, K).contiguous()
    if stride_b_batch == -1:
        b_in = b.reshape(-1, K, N).contiguous()
        stride_b_batch = b_in.stride(0)
    
    batch_size = a_in.shape[0]

    c_out = torch.empty((batch_size, M, N), device=a.device, dtype=torch.float32)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), 
        batch_size
    )

    custom_accum_gemm_kernel[grid](
        a_in, b_in, c_out,
        M, N, K,
        a_in.stride(1),
        a_in.stride(2),
        b_in.stride(0) if b.ndim == 2 else b_in.stride(1),
        b_in.stride(1) if b.ndim == 2 else b_in.stride(2),
        c_out.stride(1),
        c_out.stride(2),
        a_in.stride(0),
        stride_b_batch,
        c_out.stride(0),
        m_bits=m_bits,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=block_size_k, 
    )

    return c_out.view(*batch_dims, M, N)
