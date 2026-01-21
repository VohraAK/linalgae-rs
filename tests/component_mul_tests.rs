use linalgae_rs::{matrix, core::matrix::Matrix};
use num_traits::{Zero, One, Num};
use std::fmt::Debug;

// -------- Generic Test Functions --------

fn test_component_mul_basic<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8>,
{
    let a = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    let b = matrix![[T::from(2), T::from(3)], [T::from(4), T::from(5)]];
    let c = a.component_mul(&b);
    assert_eq!(c.as_slice(), &[T::from(2), T::from(6), T::from(12), T::from(20)]);
}

fn test_component_mul_with_zeros<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + Zero + From<u8>,
{
    let a = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    let z = Matrix::<T>::zeroes(2, 2).unwrap();
    let c = a.component_mul(&z);
    for val in c.as_slice() {
        assert_eq!(*val, T::zero());
    }
}

fn test_component_mul_with_ones<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + One + From<u8>,
{
    let a = matrix![[T::from(5), T::from(6)], [T::from(7), T::from(8)]];
    let o = Matrix::<T>::ones(2, 2).unwrap();
    let c = a.component_mul(&o);
    assert_eq!(c.as_slice(), a.as_slice());
}

fn test_component_mul_single_element<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8>,
{
    let a = matrix![[T::from(7)]];
    let b = matrix![[T::from(3)]];
    let c = a.component_mul(&b);
    assert_eq!(c.as_slice()[0], T::from(21));
}

fn test_component_mul_negative<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<i8>,
{
    let a = matrix![[T::from(2), T::from(-3)], [T::from(-4), T::from(5)]];
    let b = matrix![[T::from(-1), T::from(2)], [T::from(3), T::from(-2)]];
    let c = a.component_mul(&b);
    assert_eq!(c.as_slice(), &[T::from(-2), T::from(-6), T::from(-12), T::from(-10)]);
}

fn test_component_mul_rectangular<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8>,
{
    let a = matrix![[T::from(1), T::from(2), T::from(3)], [T::from(4), T::from(5), T::from(6)]];
    let b = matrix![[T::from(2), T::from(2), T::from(2)], [T::from(3), T::from(3), T::from(3)]];
    let c = a.component_mul(&b);
    assert_eq!(c.as_slice(), &[T::from(2), T::from(4), T::from(6), T::from(12), T::from(15), T::from(18)]);
}

fn test_component_mul_square<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8>,
{
    let a = matrix![[T::from(1), T::from(2), T::from(3)], [T::from(4), T::from(5), T::from(6)], [T::from(7), T::from(8), T::from(9)]];
    let b = matrix![[T::from(9), T::from(8), T::from(7)], [T::from(6), T::from(5), T::from(4)], [T::from(3), T::from(2), T::from(1)]];
    let c = a.component_mul(&b);
    assert_eq!(c.as_slice(), &[T::from(9), T::from(16), T::from(21), T::from(24), T::from(25), T::from(24), T::from(21), T::from(16), T::from(9)]);
}

// -------- f64 Tests --------

#[test]
fn test_component_mul_basic_f64() {
    test_component_mul_basic::<f64>();
}

#[test]
fn test_component_mul_with_zeros_f64() {
    test_component_mul_with_zeros::<f64>();
}

#[test]
fn test_component_mul_with_ones_f64() {
    test_component_mul_with_ones::<f64>();
}

#[test]
fn test_component_mul_single_element_f64() {
    test_component_mul_single_element::<f64>();
}

#[test]
fn test_component_mul_rectangular_f64() {
    test_component_mul_rectangular::<f64>();
}

#[test]
fn test_component_mul_square_f64() {
    test_component_mul_square::<f64>();
}

#[test]
fn test_component_mul_with_infinity_f64() {
    let a = matrix![[f64::INFINITY, 1.0]];
    let b = matrix![[2.0, 3.0]];
    let c = a.component_mul(&b);
    assert_eq!(c.as_slice()[0], f64::INFINITY);
}

#[test]
fn test_component_mul_with_nan_f64() {
    let a = matrix![[f64::NAN, 1.0]];
    let b = matrix![[2.0, 3.0]];
    let c = a.component_mul(&b);
    assert!(c.as_slice()[0].is_nan());
}

// -------- f32 Tests --------

#[test]
fn test_component_mul_basic_f32() {
    test_component_mul_basic::<f32>();
}

#[test]
fn test_component_mul_with_zeros_f32() {
    test_component_mul_with_zeros::<f32>();
}

#[test]
fn test_component_mul_with_ones_f32() {
    test_component_mul_with_ones::<f32>();
}

#[test]
fn test_component_mul_single_element_f32() {
    test_component_mul_single_element::<f32>();
}

#[test]
fn test_component_mul_rectangular_f32() {
    test_component_mul_rectangular::<f32>();
}

#[test]
fn test_component_mul_square_f32() {
    test_component_mul_square::<f32>();
}

// -------- i32 Tests --------

#[test]
fn test_component_mul_basic_i32() {
    test_component_mul_basic::<i32>();
}

#[test]
fn test_component_mul_with_zeros_i32() {
    test_component_mul_with_zeros::<i32>();
}

#[test]
fn test_component_mul_with_ones_i32() {
    test_component_mul_with_ones::<i32>();
}

#[test]
fn test_component_mul_negative_i32() {
    test_component_mul_negative::<i32>();
}

#[test]
fn test_component_mul_single_element_i32() {
    test_component_mul_single_element::<i32>();
}

#[test]
fn test_component_mul_rectangular_i32() {
    test_component_mul_rectangular::<i32>();
}

#[test]
fn test_component_mul_square_i32() {
    test_component_mul_square::<i32>();
}

// -------- i64 Tests --------

#[test]
fn test_component_mul_basic_i64() {
    test_component_mul_basic::<i64>();
}

#[test]
fn test_component_mul_with_zeros_i64() {
    test_component_mul_with_zeros::<i64>();
}

#[test]
fn test_component_mul_with_ones_i64() {
    test_component_mul_with_ones::<i64>();
}

#[test]
fn test_component_mul_negative_i64() {
    test_component_mul_negative::<i64>();
}

#[test]
fn test_component_mul_single_element_i64() {
    test_component_mul_single_element::<i64>();
}

#[test]
fn test_component_mul_rectangular_i64() {
    test_component_mul_rectangular::<i64>();
}

#[test]
fn test_component_mul_square_i64() {
    test_component_mul_square::<i64>();
}

// -------- Error Tests --------

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_component_mul_dimension_mismatch_f64() {
    let a = matrix![[1.0, 2.0], [3.0, 4.0]];
    let b = matrix![[1.0, 2.0, 3.0]];
    let _ = a.component_mul(&b);
}

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_component_mul_row_mismatch_f64() {
    let a = matrix![[1.0, 2.0], [3.0, 4.0]];
    let b = matrix![[1.0, 2.0]];
    let _ = a.component_mul(&b);
}

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_component_mul_dimension_mismatch_i32() {
    let a = matrix![[1, 2], [3, 4]];
    let b = matrix![[1, 2, 3]];
    let _ = a.component_mul(&b);
}

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_component_mul_row_mismatch_i32() {
    let a = matrix![[1, 2], [3, 4]];
    let b = matrix![[1, 2]];
    let _ = a.component_mul(&b);
}
