use linalgae_rs::core::matrix::Matrix;
use num_traits::{Zero, Num};
use std::fmt::Debug;
use std::ops::{Add, Sub, Mul, Div};

// -------- Generic Test Helpers --------

fn test_matrix_addition<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8>,
    for<'a> &'a Matrix<T>: Add<&'a Matrix<T>, Output = Matrix<T>>,
{
    let a = Matrix::new(2, 2, vec![T::from(1), T::from(2), T::from(3), T::from(4)]).unwrap();
    let b = Matrix::new(2, 2, vec![T::from(5), T::from(6), T::from(7), T::from(8)]).unwrap();
    let c = &a + &b;
    assert_eq!(c.as_slice(), &[T::from(6), T::from(8), T::from(10), T::from(12)]);
}

fn test_matrix_subtraction<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8>,
    for<'a> &'a Matrix<T>: Sub<&'a Matrix<T>, Output = Matrix<T>>,
{
    let a = Matrix::new(2, 2, vec![T::from(5), T::from(6), T::from(7), T::from(8)]).unwrap();
    let b = Matrix::new(2, 2, vec![T::from(1), T::from(2), T::from(3), T::from(4)]).unwrap();
    let c = &a - &b;
    assert_eq!(c.as_slice(), &[T::from(4), T::from(4), T::from(4), T::from(4)]);
}

fn test_scalar_multiplication<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8>,
    for<'a> &'a Matrix<T>: Mul<T, Output = Matrix<T>>,
{
    let a = Matrix::new(2, 2, vec![T::from(1), T::from(2), T::from(3), T::from(4)]).unwrap();
    let c = &a * T::from(2);
    assert_eq!(c.as_slice(), &[T::from(2), T::from(4), T::from(6), T::from(8)]);
}

fn test_scalar_division<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8>,
    for<'a> &'a Matrix<T>: Div<T, Output = Matrix<T>>,
{
    let a = Matrix::new(2, 2, vec![T::from(2), T::from(4), T::from(6), T::from(8)]).unwrap();
    let c = &a / T::from(2);
    assert_eq!(c.as_slice(), &[T::from(1), T::from(2), T::from(3), T::from(4)]);
}

fn test_add_to_zero_matrix<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + Zero + From<u8>,
    for<'a> &'a Matrix<T>: Add<&'a Matrix<T>, Output = Matrix<T>>,
{
    let a = Matrix::new(2, 2, vec![T::from(1), T::from(2), T::from(3), T::from(4)]).unwrap();
    let z = Matrix::<T>::zeroes(2, 2).unwrap();
    let c = &a + &z;
    assert_eq!(c.as_slice(), a.as_slice());
}

fn test_subtract_from_self<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + Zero + From<u8>,
    for<'a> &'a Matrix<T>: Sub<&'a Matrix<T>, Output = Matrix<T>>,
{
    let a = Matrix::new(2, 2, vec![T::from(1), T::from(2), T::from(3), T::from(4)]).unwrap();
    let c = &a - &a;
    for val in c.as_slice() {
        assert_eq!(*val, T::zero());
    }
}

fn test_multiply_by_zero<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + Zero + From<u8>,
    for<'a> &'a Matrix<T>: Mul<T, Output = Matrix<T>>,
{
    let a = Matrix::new(2, 2, vec![T::from(1), T::from(2), T::from(3), T::from(4)]).unwrap();
    let c = &a * T::zero();
    for val in c.as_slice() {
        assert_eq!(*val, T::zero());
    }
}

fn test_multiply_by_one<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8>,
    for<'a> &'a Matrix<T>: Mul<T, Output = Matrix<T>>,
{
    let a = Matrix::new(2, 2, vec![T::from(1), T::from(2), T::from(3), T::from(4)]).unwrap();
    let c = &a * T::one();
    assert_eq!(c.as_slice(), a.as_slice());
}

fn test_single_element_ops<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8>,
    for<'a> &'a Matrix<T>: Add<&'a Matrix<T>, Output = Matrix<T>>
        + Sub<&'a Matrix<T>, Output = Matrix<T>>
        + Mul<T, Output = Matrix<T>>
        + Div<T, Output = Matrix<T>>,
{
    let a = Matrix::new(1, 1, vec![T::from(10)]).unwrap();
    let b = Matrix::new(1, 1, vec![T::from(5)]).unwrap();

    let add = &a + &b;
    assert_eq!(add.as_slice()[0], T::from(15));

    let sub = &a - &b;
    assert_eq!(sub.as_slice()[0], T::from(5));

    let mul = &a * T::from(2);
    assert_eq!(mul.as_slice()[0], T::from(20));

    let div = &a / T::from(2);
    assert_eq!(div.as_slice()[0], T::from(5));
}

// -------- f64 Tests --------

#[test]
fn test_matrix_addition_f64() { test_matrix_addition::<f64>(); }

#[test]
fn test_matrix_subtraction_f64() { test_matrix_subtraction::<f64>(); }

#[test]
fn test_scalar_multiplication_f64() { test_scalar_multiplication::<f64>(); }

#[test]
fn test_scalar_division_f64() { test_scalar_division::<f64>(); }

#[test]
fn test_add_to_zero_matrix_f64() { test_add_to_zero_matrix::<f64>(); }

#[test]
fn test_subtract_from_self_f64() { test_subtract_from_self::<f64>(); }

#[test]
fn test_multiply_by_zero_f64() { test_multiply_by_zero::<f64>(); }

#[test]
fn test_multiply_by_one_f64() { test_multiply_by_one::<f64>(); }

#[test]
fn test_single_element_ops_f64() { test_single_element_ops::<f64>(); }

// -------- f32 Tests --------

#[test]
fn test_matrix_addition_f32() { test_matrix_addition::<f32>(); }

#[test]
fn test_matrix_subtraction_f32() { test_matrix_subtraction::<f32>(); }

#[test]
fn test_scalar_multiplication_f32() { test_scalar_multiplication::<f32>(); }

#[test]
fn test_scalar_division_f32() { test_scalar_division::<f32>(); }

#[test]
fn test_add_to_zero_matrix_f32() { test_add_to_zero_matrix::<f32>(); }

#[test]
fn test_subtract_from_self_f32() { test_subtract_from_self::<f32>(); }

#[test]
fn test_multiply_by_zero_f32() { test_multiply_by_zero::<f32>(); }

#[test]
fn test_multiply_by_one_f32() { test_multiply_by_one::<f32>(); }

#[test]
fn test_single_element_ops_f32() { test_single_element_ops::<f32>(); }

// -------- i32 Tests --------

#[test]
fn test_matrix_addition_i32() { test_matrix_addition::<i32>(); }

#[test]
fn test_matrix_subtraction_i32() { test_matrix_subtraction::<i32>(); }

#[test]
fn test_scalar_multiplication_i32() { test_scalar_multiplication::<i32>(); }

#[test]
fn test_scalar_division_i32() { test_scalar_division::<i32>(); }

#[test]
fn test_add_to_zero_matrix_i32() { test_add_to_zero_matrix::<i32>(); }

#[test]
fn test_subtract_from_self_i32() { test_subtract_from_self::<i32>(); }

#[test]
fn test_multiply_by_zero_i32() { test_multiply_by_zero::<i32>(); }

#[test]
fn test_multiply_by_one_i32() { test_multiply_by_one::<i32>(); }

#[test]
fn test_single_element_ops_i32() { test_single_element_ops::<i32>(); }

// -------- i64 Tests --------

#[test]
fn test_matrix_addition_i64() { test_matrix_addition::<i64>(); }

#[test]
fn test_matrix_subtraction_i64() { test_matrix_subtraction::<i64>(); }

#[test]
fn test_scalar_multiplication_i64() { test_scalar_multiplication::<i64>(); }

// -------- Float-specific Edge Cases --------

#[test]
fn test_add_with_infinity() {
    let a = Matrix::new(2, 2, vec![f64::INFINITY, 1.0, 2.0, 3.0]).unwrap();
    let b = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let c = &a + &b;
    assert_eq!(c.as_slice()[0], f64::INFINITY);
}

#[test]
fn test_subtract_infinity_from_infinity() {
    let a = Matrix::new(1, 1, vec![f64::INFINITY]).unwrap();
    let c = &a - &a;
    assert!(c.as_slice()[0].is_nan());
}

#[test]
fn test_multiply_infinity_by_zero() {
    let a = Matrix::new(1, 1, vec![f64::INFINITY]).unwrap();
    let c = &a * 0.0;
    assert!(c.as_slice()[0].is_nan());
}

#[test]
#[should_panic(expected = "Cannot divide by zero")]
fn test_divide_by_zero() {
    let a = Matrix::new(1, 1, vec![1.0]).unwrap();
    let _ = &a / 0.0;
}

#[test]
fn test_nan_propagation() {
    let a = Matrix::new(2, 2, vec![f64::NAN, 1.0, 2.0, 3.0]).unwrap();
    let b = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let c = &a + &b;
    assert!(c.as_slice()[0].is_nan());
}

#[test]
fn test_negative_values_subtraction() {
    let a = Matrix::new(2, 2, vec![1.0, -2.0, -3.0, 4.0]).unwrap();
    let b = Matrix::new(2, 2, vec![-1.0, 2.0, 3.0, -4.0]).unwrap();
    let c = &a - &b;
    assert_eq!(c.as_slice(), &[2.0, -4.0, -6.0, 8.0]);
}
