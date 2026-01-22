use linalgae_rs::{matrix, core::matrix::Matrix};
use num_traits::Num;
use std::fmt::Debug;

// -------- Generic Test Functions --------

fn test_sub_assign_scaled_basic<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::ops::SubAssign + std::ops::Mul<Output = T>,
{
    let mut a = matrix![[T::from(10), T::from(20)], [T::from(30), T::from(40)]];
    let b = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    a.sub_assign_scaled(&b, T::from(2));
    // a = a - b*2 = [[10-2, 20-4], [30-6, 40-8]] = [[8, 16], [24, 32]]
    assert_eq!(a.as_slice(), &[T::from(8), T::from(16), T::from(24), T::from(32)]);
}

fn test_sub_assign_scaled_zero_scalar<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::ops::SubAssign + std::ops::Mul<Output = T>,
{
    let mut a = matrix![[T::from(5), T::from(6)], [T::from(7), T::from(8)]];
    let original = a.clone();
    let b = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    a.sub_assign_scaled(&b, T::from(0));
    // a = a - b*0 = a (unchanged)
    assert_eq!(a.as_slice(), original.as_slice());
}

fn test_sub_assign_scaled_one_scalar<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::ops::SubAssign + std::ops::Mul<Output = T>,
{
    let mut a = matrix![[T::from(10), T::from(20)], [T::from(30), T::from(40)]];
    let b = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    a.sub_assign_scaled(&b, T::from(1));
    // a = a - b*1 = a - b
    assert_eq!(a.as_slice(), &[T::from(9), T::from(18), T::from(27), T::from(36)]);
}

fn test_sub_assign_scaled_negative_scalar<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<i8> + std::ops::SubAssign + std::ops::Mul<Output = T>,
{
    let mut a = matrix![[T::from(10), T::from(20)], [T::from(30), T::from(40)]];
    let b = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    a.sub_assign_scaled(&b, T::from(-2));
    // a = a - b*(-2) = a + b*2 = [[10+2, 20+4], [30+6, 40+8]] = [[12, 24], [36, 48]]
    assert_eq!(a.as_slice(), &[T::from(12), T::from(24), T::from(36), T::from(48)]);
}

fn test_sub_assign_scaled_single_element<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::ops::SubAssign + std::ops::Mul<Output = T>,
{
    let mut a = matrix![[T::from(100)]];
    let b = matrix![[T::from(10)]];
    a.sub_assign_scaled(&b, T::from(5));
    // a = 100 - 10*5 = 50
    assert_eq!(a.as_slice()[0], T::from(50));
}

fn test_sub_assign_scaled_rectangular<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::ops::SubAssign + std::ops::Mul<Output = T>,
{
    let mut a = matrix![[T::from(10), T::from(20), T::from(30)], [T::from(40), T::from(50), T::from(60)]];
    let b = matrix![[T::from(1), T::from(2), T::from(3)], [T::from(4), T::from(5), T::from(6)]];
    a.sub_assign_scaled(&b, T::from(3));
    // a = a - b*3
    assert_eq!(a.as_slice(), &[T::from(7), T::from(14), T::from(21), T::from(28), T::from(35), T::from(42)]);
}

fn test_sub_assign_scaled_with_zeros<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + From<u8> + std::ops::SubAssign + std::ops::Mul<Output = T>,
{
    let mut a = matrix![[T::from(5), T::from(6)], [T::from(7), T::from(8)]];
    let b = Matrix::<T>::zeroes(2, 2).unwrap();
    a.sub_assign_scaled(&b, T::from(10));
    // a = a - 0*10 = a (unchanged)
    assert_eq!(a.as_slice(), &[T::from(5), T::from(6), T::from(7), T::from(8)]);
}

// -------- f64 Tests --------

#[test]
fn test_sub_assign_scaled_basic_f64() {
    test_sub_assign_scaled_basic::<f64>();
}

#[test]
fn test_sub_assign_scaled_zero_scalar_f64() {
    test_sub_assign_scaled_zero_scalar::<f64>();
}

#[test]
fn test_sub_assign_scaled_one_scalar_f64() {
    test_sub_assign_scaled_one_scalar::<f64>();
}

#[test]
fn test_sub_assign_scaled_negative_scalar_f64() {
    test_sub_assign_scaled_negative_scalar::<f64>();
}

#[test]
fn test_sub_assign_scaled_single_element_f64() {
    test_sub_assign_scaled_single_element::<f64>();
}

#[test]
fn test_sub_assign_scaled_rectangular_f64() {
    test_sub_assign_scaled_rectangular::<f64>();
}

#[test]
fn test_sub_assign_scaled_with_zeros_f64() {
    test_sub_assign_scaled_with_zeros::<f64>();
}

#[test]
fn test_sub_assign_scaled_fractional_scalar_f64() {
    let mut a = matrix![[10.0, 20.0], [30.0, 40.0]];
    let b = matrix![[2.0, 4.0], [6.0, 8.0]];
    a.sub_assign_scaled(&b, 0.5);
    // a = a - b*0.5 = [[10-1, 20-2], [30-3, 40-4]] = [[9, 18], [27, 36]]
    assert_eq!(a.as_slice(), &[9.0, 18.0, 27.0, 36.0]);
}

// -------- f32 Tests --------

#[test]
fn test_sub_assign_scaled_basic_f32() {
    test_sub_assign_scaled_basic::<f32>();
}

#[test]
fn test_sub_assign_scaled_zero_scalar_f32() {
    test_sub_assign_scaled_zero_scalar::<f32>();
}

#[test]
fn test_sub_assign_scaled_one_scalar_f32() {
    test_sub_assign_scaled_one_scalar::<f32>();
}

#[test]
fn test_sub_assign_scaled_negative_scalar_f32() {
    test_sub_assign_scaled_negative_scalar::<f32>();
}

#[test]
fn test_sub_assign_scaled_rectangular_f32() {
    test_sub_assign_scaled_rectangular::<f32>();
}

// -------- i32 Tests --------

#[test]
fn test_sub_assign_scaled_basic_i32() {
    test_sub_assign_scaled_basic::<i32>();
}

#[test]
fn test_sub_assign_scaled_zero_scalar_i32() {
    test_sub_assign_scaled_zero_scalar::<i32>();
}

#[test]
fn test_sub_assign_scaled_one_scalar_i32() {
    test_sub_assign_scaled_one_scalar::<i32>();
}

#[test]
fn test_sub_assign_scaled_negative_scalar_i32() {
    test_sub_assign_scaled_negative_scalar::<i32>();
}

#[test]
fn test_sub_assign_scaled_single_element_i32() {
    test_sub_assign_scaled_single_element::<i32>();
}

#[test]
fn test_sub_assign_scaled_rectangular_i32() {
    test_sub_assign_scaled_rectangular::<i32>();
}

#[test]
fn test_sub_assign_scaled_with_zeros_i32() {
    test_sub_assign_scaled_with_zeros::<i32>();
}

// -------- i64 Tests --------

#[test]
fn test_sub_assign_scaled_basic_i64() {
    test_sub_assign_scaled_basic::<i64>();
}

#[test]
fn test_sub_assign_scaled_zero_scalar_i64() {
    test_sub_assign_scaled_zero_scalar::<i64>();
}

#[test]
fn test_sub_assign_scaled_negative_scalar_i64() {
    test_sub_assign_scaled_negative_scalar::<i64>();
}

// -------- Error Tests --------

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_sub_assign_scaled_dimension_mismatch_f64() {
    let mut a = matrix![[1.0, 2.0], [3.0, 4.0]];
    let b = matrix![[1.0, 2.0, 3.0]];
    a.sub_assign_scaled(&b, 2.0);
}

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_sub_assign_scaled_row_mismatch_f64() {
    let mut a = matrix![[1.0, 2.0], [3.0, 4.0]];
    let b = matrix![[1.0, 2.0]];
    a.sub_assign_scaled(&b, 2.0);
}

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_sub_assign_scaled_dimension_mismatch_i32() {
    let mut a = matrix![[1, 2], [3, 4]];
    let b = matrix![[1, 2, 3], [4, 5, 6]];
    a.sub_assign_scaled(&b, 2);
}

// -------- Integration Tests --------

#[test]
fn test_sub_assign_scaled_multiple_operations_f64() {
    let mut a = matrix![[100.0, 200.0]];
    let b = matrix![[10.0, 20.0]];
    let c = matrix![[5.0, 10.0]];
    
    a.sub_assign_scaled(&b, 2.0); // a = 100 - 20 = 80, 200 - 40 = 160
    a.sub_assign_scaled(&c, 4.0); // a = 80 - 20 = 60, 160 - 40 = 120
    
    assert_eq!(a.as_slice(), &[60.0, 120.0]);
}

#[test]
fn test_sub_assign_scaled_large_scalar_i32() {
    let mut a = matrix![[1000, 2000], [3000, 4000]];
    let b = matrix![[1, 2], [3, 4]];
    a.sub_assign_scaled(&b, 100);
    // a = a - b*100
    assert_eq!(a.as_slice(), &[900, 1800, 2700, 3600]);
}
