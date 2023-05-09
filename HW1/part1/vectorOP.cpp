#include "PPintrin.h"
#include <unistd.h>
// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
   __pp_vec_float x;
   __pp_vec_int y;
   __pp_vec_int count;
   __pp_vec_float result;
   __pp_vec_int zero = _pp_vset_int(0);
   __pp_vec_float zeroF = _pp_vset_float(0.f);
   __pp_vec_float oneF = _pp_vset_float(1.f);
   __pp_vec_float nineF = _pp_vset_float(9.999999f);
   __pp_vec_int one = _pp_vset_int(1);
   __pp_mask maskAll, maskIsZero, maskIsNotZero;
   __pp_mask maskCountIsNotZero;
   __pp_mask masknineIsZero;
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
      if (N>VECTOR_WIDTH) {
          if (i+ VECTOR_WIDTH>N) {
              maskAll = _pp_init_ones(N%VECTOR_WIDTH);
          }
          else {
              maskAll = _pp_init_ones();
          }
      }
      else if (N == VECTOR_WIDTH) {
          maskAll = _pp_init_ones();
      }
      else {
          maskAll = _pp_init_ones(N%VECTOR_WIDTH);
      }
      // All zeros
      maskIsZero = _pp_init_ones();
      // Load vector of values from contiguous memory addresses
      _pp_vload_float(x, values + i, maskAll); // x = values[i];
      _pp_vload_int(y, exponents + i, maskAll); // y = exponents[i];

      _pp_veq_int(maskIsZero, y, zero, maskAll); // if (y == 0) {
      _pp_vsub_float(result, oneF, zeroF, maskIsZero); // output[i] = 1.f;
      // Inverse maskIsNotZero to generate "else" mask
      maskIsNotZero = _pp_mask_not(maskIsZero); // } else {
      _pp_vsub_float(result, x, zeroF, maskIsNotZero); // result = x;
      _pp_vsub_int(count, y, one, maskIsNotZero); // count = y - 1;

      //==============while========================
      _pp_vgt_int(maskCountIsNotZero, count, zero, maskAll);
      while (_pp_cntbits(maskCountIsNotZero)>0) {
          _pp_vmult_float(result, result, x, maskCountIsNotZero);//result *= x;
          _pp_vsub_int(count, count, one, maskCountIsNotZero); // count = count - 1;
          _pp_vgt_int(maskCountIsNotZero, count, zero, maskCountIsNotZero);
      }
      //==============while========================
      masknineIsZero = _pp_init_ones();
      _pp_vlt_float(masknineIsZero, nineF, result, maskAll); // if (result > 9.999999f) {
      _pp_vsub_float(result, nineF, zeroF, masknineIsZero);//result = 9.999999f;

      _pp_vstore_float(output + i, result, maskAll); //output[i] = result;
  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{

  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //
  float sum = 0.0;
  float* output = (float*)malloc(VECTOR_WIDTH * sizeof(float));
  __pp_vec_float result = _pp_vset_float(0.f);
  __pp_vec_float x;
  __pp_mask maskAll;
  maskAll = _pp_init_ones();
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
      _pp_vload_float(x, values + i, maskAll); // x = values[i];
      _pp_vadd_float(result, result, x, maskAll);

  }
  maskAll = _pp_init_ones();
  _pp_vstore_float(output, result, maskAll); //output[i] = result;
  for (int i = 0; i < VECTOR_WIDTH; i += 1) {
      sum += output[i]; //sum = sum(result);
  }
  return sum;
}