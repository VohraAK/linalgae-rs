use linalgae_rs::{matrix, core::matrix::Matrix};
use num_traits::Num;
use std::fmt::Debug;

// -------- Generic Test Functions --------

fn test_apply_inplace_double<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8>,
{
    let m = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    let result = m.apply_inplace(|x| x + x);
    assert_eq!(result.as_slice(), &[T::from(2), T::from(4), T::from(6), T::from(8)]);
}

fn test_apply_inplace_identity<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8>,
{
    let m = matrix![[T::from(5), T::from(6)], [T::from(7), T::from(8)]];
    let original = m.clone();
    let result = m.apply_inplace(|x| x);
    assert_eq!(result.as_slice(), original.as_slice());
}

fn test_apply_inplace_square<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8>,
{
    let m = matrix![[T::from(2), T::from(3)], [T::from(4), T::from(5)]];
    let result = m.apply_inplace(|x| x * x);
    assert_eq!(result.as_slice(), &[T::from(4), T::from(9), T::from(16), T::from(25)]);
}

fn test_apply_inplace_add_constant<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8>,
{
    let m = matrix![[T::from(10), T::from(20)], [T::from(30), T::from(40)]];
    let result = m.apply_inplace(|x| x + T::from(5));
    assert_eq!(result.as_slice(), &[T::from(15), T::from(25), T::from(35), T::from(45)]);
}

fn test_apply_inplace_single_element<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8>,
{
    let m = matrix![[T::from(7)]];
    let result = m.apply_inplace(|x| x * T::from(3));
    assert_eq!(result.as_slice()[0], T::from(21));
}

fn test_apply_inplace_rectangular<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8>,
{
    let m = matrix![[T::from(1), T::from(2), T::from(3)], [T::from(4), T::from(5), T::from(6)]];
    let result = m.apply_inplace(|x| x * T::from(10));
    assert_eq!(result.as_slice(), &[T::from(10), T::from(20), T::from(30), T::from(40), T::from(50), T::from(60)]);
}

fn test_apply_inplace_zero_function<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + From<u8>,
{
    let m = matrix![[T::from(10), T::from(20)], [T::from(30), T::from(40)]];
    let result = m.apply_inplace(|_| T::zero());
    for val in result.as_slice() {
        assert_eq!(*val, T::zero());
    }
}

fn test_apply_inplace_preserves_dimensions<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8>,
{
    let m = matrix![[T::from(1), T::from(2), T::from(3)], [T::from(4), T::from(5), T::from(6)]];
    let result = m.apply_inplace(|x| x + T::from(1));
    assert_eq!(result.rows(), 2);
    assert_eq!(result.cols(), 3);
}

// -------- f64 Tests --------

#[test]
fn test_apply_inplace_double_f64() {
    test_apply_inplace_double::<f64>();
}

#[test]
fn test_apply_inplace_identity_f64() {
    test_apply_inplace_identity::<f64>();
}

#[test]
fn test_apply_inplace_square_f64() {
    test_apply_inplace_square::<f64>();
}

#[test]
fn test_apply_inplace_add_constant_f64() {
    test_apply_inplace_add_constant::<f64>();
}

#[test]
fn test_apply_inplace_single_element_f64() {
    test_apply_inplace_single_element::<f64>();
}

#[test]
fn test_apply_inplace_rectangular_f64() {
    test_apply_inplace_rectangular::<f64>();
}

#[test]
fn test_apply_inplace_zero_function_f64() {
    test_apply_inplace_zero_function::<f64>();
}

#[test]
fn test_apply_inplace_preserves_dimensions_f64() {
    test_apply_inplace_preserves_dimensions::<f64>();
}

#[test]
fn test_apply_inplace_sqrt_f64() {
    let m = matrix![[4.0f64, 9.0f64], [16.0f64, 25.0f64]];
    let result = m.apply_inplace(|x| x.sqrt());
    assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_apply_inplace_floor_f64() {
    let m = matrix![[1.7f64, 2.3f64], [3.9f64, 4.1f64]];
    let result = m.apply_inplace(|x| x.floor());
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_apply_inplace_ceil_f64() {
    let m = matrix![[1.1f64, 2.2f64], [3.3f64, 4.4f64]];
    let result = m.apply_inplace(|x| x.ceil());
    assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_apply_inplace_abs_f64() {
    let m = matrix![[-5.0f64, 3.0f64], [-2.0f64, 7.0f64]];
    let result = m.apply_inplace(|x| x.abs());
    assert_eq!(result.as_slice(), &[5.0, 3.0, 2.0, 7.0]);
}

#[test]
fn test_apply_inplace_with_capture_f64() {
    let multiplier = 2.5;
    let m = matrix![[10.0, 20.0]];
    let result = m.apply_inplace(|x| x * multiplier);
    assert_eq!(result.as_slice(), &[25.0, 50.0]);
}

// -------- f32 Tests --------

#[test]
fn test_apply_inplace_double_f32() {
    test_apply_inplace_double::<f32>();
}

#[test]
fn test_apply_inplace_identity_f32() {
    test_apply_inplace_identity::<f32>();
}

#[test]
fn test_apply_inplace_square_f32() {
    test_apply_inplace_square::<f32>();
}

#[test]
fn test_apply_inplace_add_constant_f32() {
    test_apply_inplace_add_constant::<f32>();
}

#[test]
fn test_apply_inplace_rectangular_f32() {
    test_apply_inplace_rectangular::<f32>();
}

#[test]
fn test_apply_inplace_zero_function_f32() {
    test_apply_inplace_zero_function::<f32>();
}

// -------- i32 Tests --------

#[test]
fn test_apply_inplace_double_i32() {
    test_apply_inplace_double::<i32>();
}

#[test]
fn test_apply_inplace_identity_i32() {
    test_apply_inplace_identity::<i32>();
}

#[test]
fn test_apply_inplace_square_i32() {
    test_apply_inplace_square::<i32>();
}

#[test]
fn test_apply_inplace_add_constant_i32() {
    test_apply_inplace_add_constant::<i32>();
}

#[test]
fn test_apply_inplace_single_element_i32() {
    test_apply_inplace_single_element::<i32>();
}

#[test]
fn test_apply_inplace_rectangular_i32() {
    test_apply_inplace_rectangular::<i32>();
}

#[test]
fn test_apply_inplace_zero_function_i32() {
    test_apply_inplace_zero_function::<i32>();
}

#[test]
fn test_apply_inplace_preserves_dimensions_i32() {
    test_apply_inplace_preserves_dimensions::<i32>();
}

#[test]
fn test_apply_inplace_abs_i32() {
    let m = matrix![[-5i32, 3i32], [-2i32, 7i32]];
    let result = m.apply_inplace(|x| x.abs());
    assert_eq!(result.as_slice(), &[5, 3, 2, 7]);
}

#[test]
fn test_apply_inplace_negate_i32() {
    let m = matrix![[1, -2], [3, -4]];
    let result = m.apply_inplace(|x| -x);
    assert_eq!(result.as_slice(), &[-1, 2, -3, 4]);
}

// -------- i64 Tests --------

#[test]
fn test_apply_inplace_double_i64() {
    test_apply_inplace_double::<i64>();
}

#[test]
fn test_apply_inplace_identity_i64() {
    test_apply_inplace_identity::<i64>();
}

#[test]
fn test_apply_inplace_square_i64() {
    test_apply_inplace_square::<i64>();
}

#[test]
fn test_apply_inplace_add_constant_i64() {
    test_apply_inplace_add_constant::<i64>();
}

#[test]
fn test_apply_inplace_rectangular_i64() {
    test_apply_inplace_rectangular::<i64>();
}

// -------- Comparison with apply() --------

#[test]
fn test_apply_inplace_vs_apply_same_result_f64() {
    let m1 = matrix![[1.0, 2.0], [3.0, 4.0]];
    let m2 = m1.clone();
    
    let result_apply = m1.apply(|x| x * 2.0);
    let result_inplace = m2.apply_inplace(|x| x * 2.0);
    
    assert_eq!(result_apply.as_slice(), result_inplace.as_slice());
}

#[test]
fn test_apply_inplace_vs_apply_same_result_i32() {
    let m1 = matrix![[10, 20], [30, 40]];
    let m2 = m1.clone();
    
    let result_apply = m1.apply(|x| x + 5);
    let result_inplace = m2.apply_inplace(|x| x + 5);
    
    assert_eq!(result_apply.as_slice(), result_inplace.as_slice());
}

// -------- Chaining Tests --------

#[test]
fn test_apply_inplace_chaining_f64() {
    let m = matrix![[1.0, 2.0], [3.0, 4.0]];
    let result = m
        .apply_inplace(|x| x * 2.0)      // [2, 4, 6, 8]
        .apply_inplace(|x| x + 10.0)     // [12, 14, 16, 18]
        .apply_inplace(|x| x / 2.0);     // [6, 7, 8, 9]
    assert_eq!(result.as_slice(), &[6.0, 7.0, 8.0, 9.0]);
}

#[test]
fn test_apply_inplace_chaining_i32() {
    let m = matrix![[10, 20], [30, 40]];
    let result = m
        .apply_inplace(|x| x / 10)
        .apply_inplace(|x| x * 5)
        .apply_inplace(|x| x + 1);
    assert_eq!(result.as_slice(), &[6, 11, 16, 21]);
}

// -------- Large Matrix Test --------

#[test]
fn test_apply_inplace_large_matrix_f64() {
    let m = Matrix::<f64>::ones(10, 10).unwrap();
    let result = m.apply_inplace(|x| x * 42.0);
    assert_eq!(result.as_slice().len(), 100);
    for val in result.as_slice() {
        assert_eq!(*val, 42.0);
    }
}
