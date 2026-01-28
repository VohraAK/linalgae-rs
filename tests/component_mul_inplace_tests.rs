use linalgae_rs::{matrix, core::matrix::Matrix};
use num_traits::Num;
use std::fmt::Debug;

// -------- Generic Test Functions --------

fn test_component_mul_inplace_basic<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::ops::MulAssign,
{
    let a = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    let b = matrix![[T::from(2), T::from(3)], [T::from(4), T::from(5)]];
    let result = a.component_mul_inplace(&b);
    assert_eq!(result.as_slice(), &[T::from(2), T::from(6), T::from(12), T::from(20)]);
}

fn test_component_mul_inplace_with_zeros<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + From<u8> + std::ops::MulAssign,
{
    let a = matrix![[T::from(5), T::from(6)], [T::from(7), T::from(8)]];
    let z = Matrix::<T>::zeroes(2, 2).unwrap();
    let result = a.component_mul_inplace(&z);
    for val in result.as_slice() {
        assert_eq!(*val, T::zero());
    }
}

fn test_component_mul_inplace_with_ones<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + num_traits::One + From<u8> + std::ops::MulAssign,
{
    let a = matrix![[T::from(5), T::from(6)], [T::from(7), T::from(8)]];
    let original = a.clone();
    let o = Matrix::<T>::ones(2, 2).unwrap();
    let result = a.component_mul_inplace(&o);
    assert_eq!(result.as_slice(), original.as_slice());
}

fn test_component_mul_inplace_single_element<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::ops::MulAssign,
{
    let a = matrix![[T::from(7)]];
    let b = matrix![[T::from(3)]];
    let result = a.component_mul_inplace(&b);
    assert_eq!(result.as_slice()[0], T::from(21));
}

fn test_component_mul_inplace_rectangular<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::ops::MulAssign,
{
    let a = matrix![[T::from(1), T::from(2), T::from(3)], [T::from(4), T::from(5), T::from(6)]];
    let b = matrix![[T::from(2), T::from(2), T::from(2)], [T::from(3), T::from(3), T::from(3)]];
    let result = a.component_mul_inplace(&b);
    assert_eq!(result.as_slice(), &[T::from(2), T::from(4), T::from(6), T::from(12), T::from(15), T::from(18)]);
}

fn test_component_mul_inplace_square<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::ops::MulAssign,
{
    let a = matrix![[T::from(1), T::from(2), T::from(3)], [T::from(4), T::from(5), T::from(6)], [T::from(7), T::from(8), T::from(9)]];
    let b = matrix![[T::from(9), T::from(8), T::from(7)], [T::from(6), T::from(5), T::from(4)], [T::from(3), T::from(2), T::from(1)]];
    let result = a.component_mul_inplace(&b);
    assert_eq!(result.as_slice(), &[T::from(9), T::from(16), T::from(21), T::from(24), T::from(25), T::from(24), T::from(21), T::from(16), T::from(9)]);
}

fn test_component_mul_inplace_preserves_dimensions<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + From<u8> + std::ops::MulAssign,
{
    let a = matrix![[T::from(1), T::from(2), T::from(3)], [T::from(4), T::from(5), T::from(6)]];
    let b = matrix![[T::from(1), T::from(1), T::from(1)], [T::from(1), T::from(1), T::from(1)]];
    let result = a.component_mul_inplace(&b);
    assert_eq!(result.rows(), 2);
    assert_eq!(result.cols(), 3);
}

// -------- f64 Tests --------

#[test]
fn test_component_mul_inplace_basic_f64() {
    test_component_mul_inplace_basic::<f64>();
}

#[test]
fn test_component_mul_inplace_with_zeros_f64() {
    test_component_mul_inplace_with_zeros::<f64>();
}

#[test]
fn test_component_mul_inplace_with_ones_f64() {
    test_component_mul_inplace_with_ones::<f64>();
}

#[test]
fn test_component_mul_inplace_single_element_f64() {
    test_component_mul_inplace_single_element::<f64>();
}

#[test]
fn test_component_mul_inplace_rectangular_f64() {
    test_component_mul_inplace_rectangular::<f64>();
}

#[test]
fn test_component_mul_inplace_square_f64() {
    test_component_mul_inplace_square::<f64>();
}

#[test]
fn test_component_mul_inplace_preserves_dimensions_f64() {
    test_component_mul_inplace_preserves_dimensions::<f64>();
}

#[test]
fn test_component_mul_inplace_with_fractional_f64() {
    let a = matrix![[10.0, 20.0], [30.0, 40.0]];
    let b = matrix![[0.5, 0.25], [0.1, 2.0]];
    let result = a.component_mul_inplace(&b);
    assert_eq!(result.as_slice(), &[5.0, 5.0, 3.0, 80.0]);
}

#[test]
fn test_component_mul_inplace_with_negative_f64() {
    let a = matrix![[2.0, -3.0], [-4.0, 5.0]];
    let b = matrix![[-1.0, 2.0], [3.0, -2.0]];
    let result = a.component_mul_inplace(&b);
    assert_eq!(result.as_slice(), &[-2.0, -6.0, -12.0, -10.0]);
}

// -------- f32 Tests --------

#[test]
fn test_component_mul_inplace_basic_f32() {
    test_component_mul_inplace_basic::<f32>();
}

#[test]
fn test_component_mul_inplace_with_zeros_f32() {
    test_component_mul_inplace_with_zeros::<f32>();
}

#[test]
fn test_component_mul_inplace_with_ones_f32() {
    test_component_mul_inplace_with_ones::<f32>();
}

#[test]
fn test_component_mul_inplace_rectangular_f32() {
    test_component_mul_inplace_rectangular::<f32>();
}

#[test]
fn test_component_mul_inplace_square_f32() {
    test_component_mul_inplace_square::<f32>();
}

// -------- i32 Tests --------

#[test]
fn test_component_mul_inplace_basic_i32() {
    test_component_mul_inplace_basic::<i32>();
}

#[test]
fn test_component_mul_inplace_with_zeros_i32() {
    test_component_mul_inplace_with_zeros::<i32>();
}

#[test]
fn test_component_mul_inplace_with_ones_i32() {
    test_component_mul_inplace_with_ones::<i32>();
}

#[test]
fn test_component_mul_inplace_single_element_i32() {
    test_component_mul_inplace_single_element::<i32>();
}

#[test]
fn test_component_mul_inplace_rectangular_i32() {
    test_component_mul_inplace_rectangular::<i32>();
}

#[test]
fn test_component_mul_inplace_square_i32() {
    test_component_mul_inplace_square::<i32>();
}

#[test]
fn test_component_mul_inplace_preserves_dimensions_i32() {
    test_component_mul_inplace_preserves_dimensions::<i32>();
}

#[test]
fn test_component_mul_inplace_with_negative_i32() {
    let a = matrix![[2, -3], [-4, 5]];
    let b = matrix![[-1, 2], [3, -2]];
    let result = a.component_mul_inplace(&b);
    assert_eq!(result.as_slice(), &[-2, -6, -12, -10]);
}

// -------- i64 Tests --------

#[test]
fn test_component_mul_inplace_basic_i64() {
    test_component_mul_inplace_basic::<i64>();
}

#[test]
fn test_component_mul_inplace_with_zeros_i64() {
    test_component_mul_inplace_with_zeros::<i64>();
}

#[test]
fn test_component_mul_inplace_with_ones_i64() {
    test_component_mul_inplace_with_ones::<i64>();
}

#[test]
fn test_component_mul_inplace_single_element_i64() {
    test_component_mul_inplace_single_element::<i64>();
}

#[test]
fn test_component_mul_inplace_rectangular_i64() {
    test_component_mul_inplace_rectangular::<i64>();
}

#[test]
fn test_component_mul_inplace_square_i64() {
    test_component_mul_inplace_square::<i64>();
}

// -------- Error Tests --------

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_component_mul_inplace_dimension_mismatch_f64() {
    let a = matrix![[1.0, 2.0], [3.0, 4.0]];
    let b = matrix![[1.0, 2.0, 3.0]];
    let _ = a.component_mul_inplace(&b);
}

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_component_mul_inplace_row_mismatch_f64() {
    let a = matrix![[1.0, 2.0], [3.0, 4.0]];
    let b = matrix![[1.0, 2.0]];
    let _ = a.component_mul_inplace(&b);
}

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_component_mul_inplace_dimension_mismatch_i32() {
    let a = matrix![[1, 2], [3, 4]];
    let b = matrix![[1, 2, 3], [4, 5, 6]];
    let _ = a.component_mul_inplace(&b);
}

// -------- Comparison with component_mul() --------

#[test]
fn test_component_mul_inplace_vs_component_mul_same_result_f64() {
    let a1 = matrix![[1.0, 2.0], [3.0, 4.0]];
    let a2 = a1.clone();
    let b = matrix![[5.0, 6.0], [7.0, 8.0]];
    
    let result_normal = a1.component_mul(&b);
    let result_inplace = a2.component_mul_inplace(&b);
    
    assert_eq!(result_normal.as_slice(), result_inplace.as_slice());
}

#[test]
fn test_component_mul_inplace_vs_component_mul_same_result_i32() {
    let a1 = matrix![[10, 20], [30, 40]];
    let a2 = a1.clone();
    let b = matrix![[2, 3], [4, 5]];
    
    let result_normal = a1.component_mul(&b);
    let result_inplace = a2.component_mul_inplace(&b);
    
    assert_eq!(result_normal.as_slice(), result_inplace.as_slice());
}

// -------- Chaining Tests --------

#[test]
fn test_component_mul_inplace_chaining_f64() {
    let m = matrix![[2.0, 3.0], [4.0, 5.0]];
    let multiplier1 = matrix![[2.0, 2.0], [2.0, 2.0]];
    let multiplier2 = matrix![[3.0, 3.0], [3.0, 3.0]];
    
    let result = m
        .component_mul_inplace(&multiplier1)
        .component_mul_inplace(&multiplier2);
    
    // 2*2*3=12, 3*2*3=18, 4*2*3=24, 5*2*3=30
    assert_eq!(result.as_slice(), &[12.0, 18.0, 24.0, 30.0]);
}

#[test]
fn test_component_mul_inplace_chaining_i32() {
    let m = matrix![[10, 20], [30, 40]];
    let mul1 = matrix![[2, 2], [2, 2]];
    let mul2 = matrix![[5, 5], [5, 5]];
    
    let result = m
        .component_mul_inplace(&mul1)
        .component_mul_inplace(&mul2);
    
    // 10*2*5=100, 20*2*5=200, 30*2*5=300, 40*2*5=400
    assert_eq!(result.as_slice(), &[100, 200, 300, 400]);
}

// -------- Self Multiplication --------

#[test]
fn test_component_mul_inplace_self_square_f64() {
    let m = matrix![[2.0, 3.0], [4.0, 5.0]];
    let m_clone = m.clone();
    let result = m.component_mul_inplace(&m_clone);
    // Squares each element
    assert_eq!(result.as_slice(), &[4.0, 9.0, 16.0, 25.0]);
}

#[test]
fn test_component_mul_inplace_self_square_i32() {
    let m = matrix![[2, 3], [4, 5]];
    let m_clone = m.clone();
    let result = m.component_mul_inplace(&m_clone);
    assert_eq!(result.as_slice(), &[4, 9, 16, 25]);
}

// -------- Large Matrix Test --------

#[test]
fn test_component_mul_inplace_large_matrix_f64() {
    let a = Matrix::<f64>::ones(10, 10).unwrap();
    let b = Matrix::<f64>::full(10, 10, 5.0).unwrap();
    let result = a.component_mul_inplace(&b);
    assert_eq!(result.as_slice().len(), 100);
    for val in result.as_slice() {
        assert_eq!(*val, 5.0);
    }
}

// -------- Combined with apply_inplace --------

#[test]
fn test_component_mul_inplace_with_apply_inplace_f64() {
    let a = matrix![[2.0, 3.0], [4.0, 5.0]];
    let b = matrix![[2.0, 2.0], [2.0, 2.0]];
    
    let result = a
        .component_mul_inplace(&b)
        .apply_inplace(|x| x + 10.0);
    
    // (2*2)+10=14, (3*2)+10=16, (4*2)+10=18, (5*2)+10=20
    assert_eq!(result.as_slice(), &[14.0, 16.0, 18.0, 20.0]);
}
