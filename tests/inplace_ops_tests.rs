use linalgae_rs::{matrix, core::matrix::Matrix};
use num_traits::Num;
use std::fmt::Debug;

// -------- AddAssign Tests --------

fn test_add_assign_basic<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::ops::AddAssign,
{
    let mut a = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    let b = matrix![[T::from(5), T::from(6)], [T::from(7), T::from(8)]];
    a += &b;
    assert_eq!(a.as_slice(), &[T::from(6), T::from(8), T::from(10), T::from(12)]);
}

fn test_add_assign_with_zeros<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + From<u8> + std::ops::AddAssign,
{
    let mut a = matrix![[T::from(5), T::from(6)], [T::from(7), T::from(8)]];
    let original = a.clone();
    let z = Matrix::<T>::zeroes(2, 2).unwrap();
    a += &z;
    assert_eq!(a.as_slice(), original.as_slice());
}

fn test_add_assign_single_element<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::ops::AddAssign,
{
    let mut a = matrix![[T::from(10)]];
    let b = matrix![[T::from(25)]];
    a += &b;
    assert_eq!(a.as_slice()[0], T::from(35));
}

fn test_add_assign_rectangular<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::ops::AddAssign,
{
    let mut a = matrix![[T::from(1), T::from(2), T::from(3)], [T::from(4), T::from(5), T::from(6)]];
    let b = matrix![[T::from(10), T::from(20), T::from(30)], [T::from(40), T::from(50), T::from(60)]];
    a += &b;
    assert_eq!(a.as_slice(), &[T::from(11), T::from(22), T::from(33), T::from(44), T::from(55), T::from(66)]);
}

fn test_add_assign_multiple_times<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::ops::AddAssign,
{
    let mut a = matrix![[T::from(1), T::from(2)]];
    let b = matrix![[T::from(3), T::from(4)]];
    a += &b;
    a += &b;
    a += &b;
    assert_eq!(a.as_slice(), &[T::from(10), T::from(14)]);
}

// -------- SubAssign Tests --------

fn test_sub_assign_basic<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::ops::SubAssign,
{
    let mut a = matrix![[T::from(10), T::from(20)], [T::from(30), T::from(40)]];
    let b = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    a -= &b;
    assert_eq!(a.as_slice(), &[T::from(9), T::from(18), T::from(27), T::from(36)]);
}

fn test_sub_assign_with_zeros<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + From<u8> + std::ops::SubAssign,
{
    let mut a = matrix![[T::from(5), T::from(6)], [T::from(7), T::from(8)]];
    let original = a.clone();
    let z = Matrix::<T>::zeroes(2, 2).unwrap();
    a -= &z;
    assert_eq!(a.as_slice(), original.as_slice());
}

fn test_sub_assign_single_element<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::ops::SubAssign,
{
    let mut a = matrix![[T::from(100)]];
    let b = matrix![[T::from(25)]];
    a -= &b;
    assert_eq!(a.as_slice()[0], T::from(75));
}

fn test_sub_assign_rectangular<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::ops::SubAssign,
{
    let mut a = matrix![[T::from(50), T::from(60), T::from(70)], [T::from(80), T::from(90), T::from(100)]];
    let b = matrix![[T::from(10), T::from(20), T::from(30)], [T::from(40), T::from(50), T::from(60)]];
    a -= &b;
    assert_eq!(a.as_slice(), &[T::from(40), T::from(40), T::from(40), T::from(40), T::from(40), T::from(40)]);
}

fn test_sub_assign_from_self<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + From<u8> + std::ops::SubAssign,
{
    let mut a = matrix![[T::from(5), T::from(6)], [T::from(7), T::from(8)]];
    let b = a.clone();
    a -= &b;
    for val in a.as_slice() {
        assert_eq!(*val, T::zero());
    }
}

// -------- f64 Tests --------

#[test]
fn test_add_assign_basic_f64() {
    test_add_assign_basic::<f64>();
}

#[test]
fn test_add_assign_with_zeros_f64() {
    test_add_assign_with_zeros::<f64>();
}

#[test]
fn test_add_assign_single_element_f64() {
    test_add_assign_single_element::<f64>();
}

#[test]
fn test_add_assign_rectangular_f64() {
    test_add_assign_rectangular::<f64>();
}

#[test]
fn test_add_assign_multiple_times_f64() {
    test_add_assign_multiple_times::<f64>();
}

#[test]
fn test_sub_assign_basic_f64() {
    test_sub_assign_basic::<f64>();
}

#[test]
fn test_sub_assign_with_zeros_f64() {
    test_sub_assign_with_zeros::<f64>();
}

#[test]
fn test_sub_assign_single_element_f64() {
    test_sub_assign_single_element::<f64>();
}

#[test]
fn test_sub_assign_rectangular_f64() {
    test_sub_assign_rectangular::<f64>();
}

#[test]
fn test_sub_assign_from_self_f64() {
    test_sub_assign_from_self::<f64>();
}

#[test]
fn test_add_assign_with_negative_values_f64() {
    let mut a = matrix![[5.0, -3.0], [-2.0, 4.0]];
    let b = matrix![[-1.0, 2.0], [3.0, -4.0]];
    a += &b;
    assert_eq!(a.as_slice(), &[4.0, -1.0, 1.0, 0.0]);
}

#[test]
fn test_sub_assign_with_negative_values_f64() {
    let mut a = matrix![[5.0, -3.0], [-2.0, 4.0]];
    let b = matrix![[-1.0, 2.0], [3.0, -4.0]];
    a -= &b;
    assert_eq!(a.as_slice(), &[6.0, -5.0, -5.0, 8.0]);
}

// -------- f32 Tests --------

#[test]
fn test_add_assign_basic_f32() {
    test_add_assign_basic::<f32>();
}

#[test]
fn test_add_assign_with_zeros_f32() {
    test_add_assign_with_zeros::<f32>();
}

#[test]
fn test_add_assign_rectangular_f32() {
    test_add_assign_rectangular::<f32>();
}

#[test]
fn test_sub_assign_basic_f32() {
    test_sub_assign_basic::<f32>();
}

#[test]
fn test_sub_assign_with_zeros_f32() {
    test_sub_assign_with_zeros::<f32>();
}

#[test]
fn test_sub_assign_from_self_f32() {
    test_sub_assign_from_self::<f32>();
}

// -------- i32 Tests --------

#[test]
fn test_add_assign_basic_i32() {
    test_add_assign_basic::<i32>();
}

#[test]
fn test_add_assign_with_zeros_i32() {
    test_add_assign_with_zeros::<i32>();
}

#[test]
fn test_add_assign_single_element_i32() {
    test_add_assign_single_element::<i32>();
}

#[test]
fn test_add_assign_rectangular_i32() {
    test_add_assign_rectangular::<i32>();
}

#[test]
fn test_add_assign_multiple_times_i32() {
    test_add_assign_multiple_times::<i32>();
}

#[test]
fn test_sub_assign_basic_i32() {
    test_sub_assign_basic::<i32>();
}

#[test]
fn test_sub_assign_with_zeros_i32() {
    test_sub_assign_with_zeros::<i32>();
}

#[test]
fn test_sub_assign_single_element_i32() {
    test_sub_assign_single_element::<i32>();
}

#[test]
fn test_sub_assign_rectangular_i32() {
    test_sub_assign_rectangular::<i32>();
}

#[test]
fn test_sub_assign_from_self_i32() {
    test_sub_assign_from_self::<i32>();
}

// -------- i64 Tests --------

#[test]
fn test_add_assign_basic_i64() {
    test_add_assign_basic::<i64>();
}

#[test]
fn test_add_assign_with_zeros_i64() {
    test_add_assign_with_zeros::<i64>();
}

#[test]
fn test_add_assign_multiple_times_i64() {
    test_add_assign_multiple_times::<i64>();
}

#[test]
fn test_sub_assign_basic_i64() {
    test_sub_assign_basic::<i64>();
}

#[test]
fn test_sub_assign_with_zeros_i64() {
    test_sub_assign_with_zeros::<i64>();
}

#[test]
fn test_sub_assign_from_self_i64() {
    test_sub_assign_from_self::<i64>();
}

// -------- Error Tests --------

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_add_assign_dimension_mismatch_f64() {
    let mut a = matrix![[1.0, 2.0], [3.0, 4.0]];
    let b = matrix![[1.0, 2.0, 3.0]];
    a += &b;
}

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_add_assign_row_mismatch_f64() {
    let mut a = matrix![[1.0, 2.0], [3.0, 4.0]];
    let b = matrix![[1.0, 2.0]];
    a += &b;
}

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_sub_assign_dimension_mismatch_f64() {
    let mut a = matrix![[1.0, 2.0], [3.0, 4.0]];
    let b = matrix![[1.0, 2.0, 3.0]];
    a -= &b;
}

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_sub_assign_row_mismatch_f64() {
    let mut a = matrix![[1.0, 2.0], [3.0, 4.0]];
    let b = matrix![[1.0, 2.0]];
    a -= &b;
}

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_add_assign_dimension_mismatch_i32() {
    let mut a = matrix![[1, 2], [3, 4]];
    let b = matrix![[1, 2, 3], [4, 5, 6]];
    a += &b;
}

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_sub_assign_dimension_mismatch_i32() {
    let mut a = matrix![[1, 2], [3, 4]];
    let b = matrix![[1, 2, 3], [4, 5, 6]];
    a -= &b;
}

// -------- Combined Operations Tests --------

#[test]
fn test_add_sub_assign_combo_f64() {
    let mut a = matrix![[10.0, 20.0]];
    let b = matrix![[5.0, 10.0]];
    let c = matrix![[2.0, 4.0]];
    
    a += &b; // [15, 30]
    a -= &c; // [13, 26]
    a += &b; // [18, 36]
    
    assert_eq!(a.as_slice(), &[18.0, 36.0]);
}

#[test]
fn test_add_sub_assign_combo_i32() {
    let mut a = matrix![[100, 200], [300, 400]];
    let b = matrix![[10, 20], [30, 40]];
    
    a += &b; // [110, 220], [330, 440]
    a -= &b; // [100, 200], [300, 400]
    a -= &b; // [90, 180], [270, 360]
    
    assert_eq!(a.as_slice(), &[90, 180, 270, 360]);
}

#[test]
fn test_inplace_operations_preserve_dimensions() {
    let mut a = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let b = matrix![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
    
    a += &b;
    assert_eq!(a.rows(), 3);
    assert_eq!(a.cols(), 3);
    
    a -= &b;
    assert_eq!(a.rows(), 3);
    assert_eq!(a.cols(), 3);
}
