import numpy as np
from bitstring import BitArray
        
import torch
import warnings

def int2bin(input, num_bits):
    '''
    convert the signed integer value into unsigned integer (2's complement equivalently).
    '''
    output = input.clone()
    output[input.lt(0)] = 2**num_bits + output[input.lt(0)]
    return output


def bin2int(input, num_bits):
    '''
    convert the unsigned integer (2's complement equivantly) back to the signed integer format
    with the bitwise operations. Note that, in order to perform the bitwise operation, the input
    tensor has to be in the integer format.
    '''
    mask = 2**(num_bits - 1) - 1
    output = -(input & ~mask) + (input & mask)
    return output


def flipFloat(val, bit=None):
    # Cast float to BitArray and flip (invert) random bit 0-31

    faultValue = BitArray(float=val, length=32)
    if bit == None:
        bit = np.random.randint(0, faultValue.len)
    faultValue.invert(bit)
    return faultValue.float, bit

def flipInt(val, size, bit=None):
    # Cast integer to BitArray and flip (invert) random bit 0-N
    val = int(val)
    faultValue = BitArray(int=val, length=size)
    if bit == None:
        bit = np.random.randint(0, faultValue.len)
    faultValue.invert(bit)
    return faultValue.int, bit


def bitFlip(value, size, bit=None,  quantized=False):
    if quantized:
        return flipInt(value, size, bit)
    else:
        return flipFloat(value, bit)



def bit2float(b, num_e_bits=8, num_m_bits=23, bias=127.):
  """Turn input tensor into float.
      Args:
          b : binary tensor. The last dimension of this tensor should be the
          the one the binary is at.
          num_e_bits : Number of exponent bits. Default: 8.
          num_m_bits : Number of mantissa bits. Default: 23.
          bias : Exponent bias/ zero offset. Default: 127.
      Returns:
          Tensor: Float tensor. Reduces last dimension.
  """
  expected_last_dim = num_m_bits + num_e_bits + 1
  assert b.shape[-1] == expected_last_dim, "Binary tensors last dimension " \
                                           "should be {}, not {}.".format(
    expected_last_dim, b.shape[-1])

  # check if we got the right type
  dtype = torch.float32
  if expected_last_dim > 32: dtype = torch.float64
  if expected_last_dim > 64:
    warnings.warn("pytorch can not process floats larger than 64 bits, keep"
                  " this in mind. Your result will be not exact.")

  s = torch.index_select(b, -1, torch.arange(0, 1).cuda())
  e = torch.index_select(b, -1, torch.arange(1, 1 + num_e_bits).cuda())
  m = torch.index_select(b, -1, torch.arange(1 + num_e_bits,
                                             1 + num_e_bits + num_m_bits).cuda())

  # SIGN BIT
  out = ((-1) ** s).squeeze(-1).type(dtype)
  # EXPONENT BIT
  exponents = -torch.arange(-(num_e_bits - 1.), 1.).cuda()
  exponents = exponents.repeat(b.shape[:-1] + (1,))
  e_decimal = torch.sum(e * 2 ** exponents, dim=-1) - bias
  e_decimal = torch.where(e_decimal>127, torch.full_like(e_decimal, 127), e_decimal)
  e_decimal = torch.where(e_decimal<-127, torch.full_like(e_decimal, -127), e_decimal)  
  out *= 2 ** e_decimal
  # MANTISSA
  matissa = (torch.Tensor([2.]) ** (
    -torch.arange(1., num_m_bits + 1.))).repeat(
    m.shape[:-1] + (1,)).cuda()
  w = torch.sum(m * matissa, dim=-1)
  w = torch.where(w==0, torch.full_like(w, -1), w)
  mn = 1. + w
  out *= mn
  
  return out


def float2bit(f, num_e_bits=8, num_m_bits=23, bias=127., dtype=torch.float32):
  """Turn input tensor into binary.
      Args:
          f : float tensor.
          num_e_bits : Number of exponent bits. Default: 8.
          num_m_bits : Number of mantissa bits. Default: 23.
          bias : Exponent bias/ zero offset. Default: 127.
          dtype : This is the actual type of the tensor that is going to be
          returned. Default: torch.float32.
      Returns:
          Tensor: Binary tensor. Adds last dimension to original tensor for
          bits.
  """
  f = torch.where(torch.isnan(f), torch.full_like(f, 0), f)
  s = torch.sign(f)
  s = torch.where(s==0, torch.full_like(s, 1), s)
  f = torch.abs(f)
  # turn sign into sign-bit
  s = (s * (-1) + 1.) * 0.5
  s = s.unsqueeze(-1)
  

  ## EXPONENT BIT
  x=torch.log2(f)
  e_scientific = torch.floor(x)
  e_decimal = e_scientific + bias
  e = integer2bit(e_decimal, num_bits=num_e_bits)
  e = torch.where(torch.isnan(e), torch.full_like(e, 0), e)
  e_scientific = torch.where(torch.isinf(e_scientific), torch.full_like(e_scientific, 0), e_scientific)
  e_scientific = torch.where(e_scientific<-104, torch.full_like(e_scientific, -104), e_scientific)
  
  ## MANTISSA
  m1 = integer2bit(f - f % 1, num_bits=num_e_bits)
  m2 = remainder2bit(f % 1, num_bits=bias)
  m = torch.cat([m1, m2], dim=-1)
  
  dtype = f.type()
  idx = torch.arange(num_m_bits).unsqueeze(0).type(dtype) \
        + (8. - e_scientific).unsqueeze(-1)
  idx = torch.abs(idx)
  idx = idx.long()
  m = torch.gather(m, dim=-1, index=idx)

  return torch.cat([s, e, m], dim=-1).type(dtype)


def remainder2bit(remainder, num_bits=127):
  """Turn a tensor with remainders (floats < 1) to mantissa bits.
      Args:
          remainder : torch.Tensor, tensor with remainders
          num_bits : Number of bits to specify the precision. Default: 127.
      Returns:
          Tensor: Binary tensor. Adds last dimension to original tensor for
          bits.
  """
  dtype = remainder.type()
  exponent_bits = torch.arange(num_bits).type(dtype)
  exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
  out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
  return torch.floor(2 * out)


def integer2bit(integer, num_bits=8):
  """Turn integer tensor to binary representation.
      Args:
          integer : torch.Tensor, tensor with integers
          num_bits : Number of bits to specify the precision. Default: 8.
      Returns:
          Tensor: Binary tensor. Adds last dimension to original tensor for
          bits.
  """
  dtype = integer.type()
  exponent_bits = -torch.arange(-(num_bits - 1), 1).type(dtype)
  exponent_bits = exponent_bits.repeat(integer.shape + (1,))
  out = integer.unsqueeze(-1) / 2 ** exponent_bits
  return (out - (out % 1)) % 2