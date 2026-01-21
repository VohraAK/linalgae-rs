use linalgae_rs::{matrix, core::matrix::Matrix};
use num_traits::{Zero, One, Num};
use std::fmt::Debug;

// -------- Generic Test Functions --------

fn test_matmul_basic<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + From<u8> + std::ops::AddAssign,
{
    let a = matrix![
        [T::from(1), T::from(2), T::from(3)],
        [T::from(4), T::from(5), T::from(6)]
    ];
    let b = matrix![
        [T::from(7), T::from(8)],
        [T::from(9), T::from(10)],
        [T::from(11), T::from(12)]
    ];
    let c = a.matmul(&b);
    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 2);
    assert_eq!(c.as_slice(), &[T::from(58), T::from(64), T::from(139), T::from(154)]);
}

fn test_matmul_identity<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + Zero + One + From<u8> + std::ops::AddAssign,
{
    let a = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    let i = Matrix::<T>::identity(2).unwrap();
    let c = a.matmul(&i);
    assert_eq!(c.as_slice(), a.as_slice());
}

fn test_matmul_square<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + From<u8> + std::ops::AddAssign,
{
    let a = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    let b = matrix![[T::from(2), T::from(0)], [T::from(1), T::from(2)]];
    let c = a.matmul(&b);
    // [1*2+2*1, 1*0+2*2] = [4, 4]
    // [3*2+4*1, 3*0+4*2] = [10, 8]
    assert_eq!(c.as_slice(), &[T::from(4), T::from(4), T::from(10), T::from(8)]);
}

fn test_matmul_single_element<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + From<u8> + std::ops::AddAssign,
{
    let a = matrix![[T::from(5)]];
    let b = matrix![[T::from(3)]];
    let c = a.matmul(&b);
    assert_eq!(c.as_slice()[0], T::from(15));
}

fn test_matmul_with_zeros<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + Zero + From<u8> + std::ops::AddAssign,
{
    let a = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    let z = Matrix::<T>::zeroes(2, 2).unwrap();
    let c = a.matmul(&z);
    for val in c.as_slice() {
        assert_eq!(*val, T::zero());
    }
}

fn test_matmul_rectangular<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + From<u8> + std::ops::AddAssign,
{
    let a = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)], [T::from(5), T::from(6)]];
    let b = matrix![[T::from(7), T::from(8), T::from(9)], [T::from(10), T::from(11), T::from(12)]];
    let c = a.matmul(&b);
    assert_eq!(c.rows(), 3);
    assert_eq!(c.cols(), 3);
}

// -------- f64 Tests --------

#[test]
fn test_matmul_basic_f64() {
    test_matmul_basic::<f64>();
}

#[test]
fn test_matmul_identity_f64() {
    test_matmul_identity::<f64>();
}

#[test]
fn test_matmul_square_f64() {
    test_matmul_square::<f64>();
}

#[test]
fn test_matmul_single_element_f64() {
    test_matmul_single_element::<f64>();
}

#[test]
fn test_matmul_with_zeros_f64() {
    test_matmul_with_zeros::<f64>();
}

#[test]
fn test_matmul_rectangular_f64() {
    test_matmul_rectangular::<f64>();
}

#[test]
fn test_matmul_with_infinity_f64() {
    let a = matrix![[f64::INFINITY, 1.0], [2.0, 3.0]];
    let b = matrix![[1.0, 2.0], [3.0, 4.0]];
    let c = a.matmul(&b);
    assert_eq!(c.as_slice()[0], f64::INFINITY);
}

// -------- f32 Tests --------

#[test]
fn test_matmul_basic_f32() {
    test_matmul_basic::<f32>();
}

#[test]
fn test_matmul_identity_f32() {
    test_matmul_identity::<f32>();
}

#[test]
fn test_matmul_square_f32() {
    test_matmul_square::<f32>();
}

#[test]
fn test_matmul_single_element_f32() {
    test_matmul_single_element::<f32>();
}

#[test]
fn test_matmul_with_zeros_f32() {
    test_matmul_with_zeros::<f32>();
}

#[test]
fn test_matmul_rectangular_f32() {
    test_matmul_rectangular::<f32>();
}

// -------- i32 Tests --------

#[test]
fn test_matmul_basic_i32() {
    test_matmul_basic::<i32>();
}

#[test]
fn test_matmul_identity_i32() {
    test_matmul_identity::<i32>();
}

#[test]
fn test_matmul_square_i32() {
    test_matmul_square::<i32>();
}

#[test]
fn test_matmul_single_element_i32() {
    test_matmul_single_element::<i32>();
}

#[test]
fn test_matmul_with_zeros_i32() {
    test_matmul_with_zeros::<i32>();
}

#[test]
fn test_matmul_rectangular_i32() {
    test_matmul_rectangular::<i32>();
}

// -------- i64 Tests --------

#[test]
fn test_matmul_basic_i64() {
    test_matmul_basic::<i64>();
}

#[test]
fn test_matmul_identity_i64() {
    test_matmul_identity::<i64>();
}

#[test]
fn test_matmul_square_i64() {
    test_matmul_square::<i64>();
}

#[test]
fn test_matmul_single_element_i64() {
    test_matmul_single_element::<i64>();
}

#[test]
fn test_matmul_with_zeros_i64() {
    test_matmul_with_zeros::<i64>();
}

#[test]
fn test_matmul_rectangular_i64() {
    test_matmul_rectangular::<i64>();
}

// -------- Error Tests --------

#[test]
#[should_panic(expected = "lhs_cols")]
fn test_matmul_dimension_mismatch_f64() {
    let a = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b = matrix![[1.0, 2.0], [3.0, 4.0]];
    let _ = a.matmul(&b);
}

#[test]
#[should_panic(expected = "lhs_cols")]
fn test_matmul_dimension_mismatch_i32() {
    let a = matrix![[1, 2, 3], [4, 5, 6]];
    let b = matrix![[1, 2], [3, 4]];
    let _ = a.matmul(&b);
}
