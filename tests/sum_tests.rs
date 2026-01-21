use linalgae_rs::{matrix, core::matrix::Matrix};
use num_traits::{Zero, One, Num};
use std::fmt::Debug;

// -------- Generic Test Functions --------

fn test_sum_basic<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::iter::Sum<T>,
{
    let m = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    let sum = m.sum();
    assert_eq!(sum, T::from(10));
}

fn test_sum_single_element<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::iter::Sum<T>,
{
    let m = matrix![[T::from(42)]];
    let sum = m.sum();
    assert_eq!(sum, T::from(42));
}

fn test_sum_zeros<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + Zero + std::iter::Sum<T>,
{
    let m = Matrix::<T>::zeroes(3, 3).unwrap();
    let sum = m.sum();
    assert_eq!(sum, T::zero());
}

fn test_sum_ones<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + One + From<u8> + std::iter::Sum<T>,
{
    let m = Matrix::<T>::ones(4, 5).unwrap();
    let sum = m.sum();
    assert_eq!(sum, T::from(20)); // 4*5 = 20
}

fn test_sum_negative<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<i8> + std::iter::Sum<T>,
{
    let m = matrix![[T::from(5), T::from(-3)], [T::from(-2), T::from(4)]];
    let sum = m.sum();
    assert_eq!(sum, T::from(4)); // 5-3-2+4 = 4
}

fn test_sum_large_matrix<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::iter::Sum<T>,
{
    let m = Matrix::<T>::full(10, 10, T::from(2)).unwrap();
    let sum = m.sum();
    assert_eq!(sum, T::from(200)); // 100 elements * 2
}

fn test_sum_rectangular<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::iter::Sum<T>,
{
    let m = matrix![[T::from(1), T::from(2), T::from(3)], [T::from(4), T::from(5), T::from(6)]];
    let sum = m.sum();
    assert_eq!(sum, T::from(21)); // 1+2+3+4+5+6 = 21
}

// -------- f64 Tests --------

#[test]
fn test_sum_basic_f64() {
    test_sum_basic::<f64>();
}

#[test]
fn test_sum_single_element_f64() {
    test_sum_single_element::<f64>();
}

#[test]
fn test_sum_zeros_f64() {
    test_sum_zeros::<f64>();
}

#[test]
fn test_sum_ones_f64() {
    test_sum_ones::<f64>();
}

#[test]
fn test_sum_large_matrix_f64() {
    test_sum_large_matrix::<f64>();
}

#[test]
fn test_sum_rectangular_f64() {
    test_sum_rectangular::<f64>();
}

#[test]
fn test_sum_with_floats_f64() {
    let m = matrix![[1.5f64, 2.3f64], [3.7f64, 4.1f64]];
    let sum = m.sum();
    assert!((sum - 11.6).abs() < 1e-10);
}

#[test]
fn test_sum_with_infinity_f64() {
    let m = matrix![[f64::INFINITY, 1.0], [2.0, 3.0]];
    let sum = m.sum();
    assert_eq!(sum, f64::INFINITY);
}

#[test]
fn test_sum_with_nan_f64() {
    let m = matrix![[f64::NAN, 1.0], [2.0, 3.0]];
    let sum = m.sum();
    assert!(sum.is_nan());
}

// -------- f32 Tests --------

#[test]
fn test_sum_basic_f32() {
    test_sum_basic::<f32>();
}

#[test]
fn test_sum_single_element_f32() {
    test_sum_single_element::<f32>();
}

#[test]
fn test_sum_zeros_f32() {
    test_sum_zeros::<f32>();
}

#[test]
fn test_sum_ones_f32() {
    test_sum_ones::<f32>();
}

#[test]
fn test_sum_large_matrix_f32() {
    test_sum_large_matrix::<f32>();
}

#[test]
fn test_sum_rectangular_f32() {
    test_sum_rectangular::<f32>();
}

#[test]
fn test_sum_with_floats_f32() {
    let m = matrix![[1.5f32, 2.3f32], [3.7f32, 4.1f32]];
    let sum = m.sum();
    assert!((sum - 11.6).abs() < 1e-6);
}

// -------- i32 Tests --------

#[test]
fn test_sum_basic_i32() {
    test_sum_basic::<i32>();
}

#[test]
fn test_sum_single_element_i32() {
    test_sum_single_element::<i32>();
}

#[test]
fn test_sum_zeros_i32() {
    test_sum_zeros::<i32>();
}

#[test]
fn test_sum_ones_i32() {
    test_sum_ones::<i32>();
}

#[test]
fn test_sum_negative_i32() {
    test_sum_negative::<i32>();
}

#[test]
fn test_sum_large_matrix_i32() {
    test_sum_large_matrix::<i32>();
}

#[test]
fn test_sum_rectangular_i32() {
    test_sum_rectangular::<i32>();
}

// -------- i64 Tests --------

#[test]
fn test_sum_basic_i64() {
    test_sum_basic::<i64>();
}

#[test]
fn test_sum_single_element_i64() {
    test_sum_single_element::<i64>();
}

#[test]
fn test_sum_zeros_i64() {
    test_sum_zeros::<i64>();
}

#[test]
fn test_sum_ones_i64() {
    test_sum_ones::<i64>();
}

#[test]
fn test_sum_negative_i64() {
    test_sum_negative::<i64>();
}

#[test]
fn test_sum_large_matrix_i64() {
    test_sum_large_matrix::<i64>();
}

#[test]
fn test_sum_rectangular_i64() {
    test_sum_rectangular::<i64>();
}
