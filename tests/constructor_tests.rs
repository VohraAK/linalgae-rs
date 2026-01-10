use linalgae_rs::core::matrix::Matrix;
use num_traits::{Zero, One, Num};
use std::fmt::Debug;

// -------- Generic Test Helpers --------

fn test_new_valid<T>()
where
    T: Num + Copy + PartialEq + Debug + From<u8>,
{
    let m = Matrix::new(2, 3, vec![T::from(1), T::from(2), T::from(3), T::from(4), T::from(5), T::from(6)]);
    assert!(m.is_ok());
    let m = m.unwrap();
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);
}

fn test_new_dimension_mismatch<T>()
where
    T: Num + Copy + From<u8>,
{
    let m = Matrix::new(2, 2, vec![T::from(1), T::from(2), T::from(3)]);
    assert!(m.is_err());
}

fn test_full<T>()
where
    T: Num + Clone + Copy + PartialEq + Debug + From<u8>,
{
    let fill = T::from(7);
    let m = Matrix::full(2, 3, fill).unwrap();
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);
    for val in m.as_slice() {
        assert_eq!(*val, fill);
    }
}

fn test_zeroes<T>()
where
    T: Num + Clone + Copy + PartialEq + Debug + Zero + Default,
{
    let m = Matrix::<T>::zeroes(3, 3).unwrap();
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 3);
    for val in m.as_slice() {
        assert_eq!(*val, T::zero());
    }
}

fn test_ones<T>()
where
    T: Num + Clone + Copy + PartialEq + Debug + One,
{
    let m = Matrix::<T>::ones(3, 2).unwrap();
    assert_eq!(m.rows(), 3);
    assert_eq!(m.cols(), 2);
    for val in m.as_slice() {
        assert_eq!(*val, T::one());
    }
}

fn test_as_slice<T>()
where
    T: Num + Copy + PartialEq + Debug + From<u8>,
{
    let data = vec![T::from(1), T::from(2), T::from(3), T::from(4)];
    let m = Matrix::new(2, 2, data.clone()).unwrap();
    assert_eq!(m.as_slice(), &data[..]);
}

fn test_as_mut_slice<T>()
where
    T: Num + Clone + Copy + PartialEq + Debug + Zero + Default + From<u8>,
{
    let mut m = Matrix::<T>::zeroes(2, 2).unwrap();
    m.as_mut_slice()[0] = T::from(5);
    m.as_mut_slice()[3] = T::from(10);
    assert_eq!(m.as_slice()[0], T::from(5));
    assert_eq!(m.as_slice()[3], T::from(10));
}

fn test_single_element<T>()
where
    T: Num + Copy + PartialEq + Debug + From<u8>,
{
    let val = T::from(42);
    let m = Matrix::new(1, 1, vec![val]).unwrap();
    assert_eq!(m.rows(), 1);
    assert_eq!(m.cols(), 1);
    assert_eq!(m.as_slice()[0], val);
}

fn test_single_row<T>()
where
    T: Num + Copy + From<u8>,
{
    let m = Matrix::new(1, 5, vec![T::from(1), T::from(2), T::from(3), T::from(4), T::from(5)]).unwrap();
    assert_eq!(m.rows(), 1);
    assert_eq!(m.cols(), 5);
}

fn test_single_column<T>()
where
    T: Num + Copy + From<u8>,
{
    let m = Matrix::new(5, 1, vec![T::from(1), T::from(2), T::from(3), T::from(4), T::from(5)]).unwrap();
    assert_eq!(m.rows(), 5);
    assert_eq!(m.cols(), 1);
}

fn test_identity<T>()
where
    T: Num + Clone + Copy + PartialEq + Debug + Zero + One,
{
    let i = Matrix::<T>::identity(3).unwrap();
    assert_eq!(i.rows(), 3);
    assert_eq!(i.cols(), 3);
    
    // Check diagonal elements are 1
    assert_eq!(i.as_slice()[0], T::one());  // (0,0)
    assert_eq!(i.as_slice()[4], T::one());  // (1,1)
    assert_eq!(i.as_slice()[8], T::one());  // (2,2)
    
    // Check off-diagonal elements are 0
    assert_eq!(i.as_slice()[1], T::zero()); // (0,1)
    assert_eq!(i.as_slice()[2], T::zero()); // (0,2)
    assert_eq!(i.as_slice()[3], T::zero()); // (1,0)
    assert_eq!(i.as_slice()[5], T::zero()); // (1,2)
    assert_eq!(i.as_slice()[6], T::zero()); // (2,0)
    assert_eq!(i.as_slice()[7], T::zero()); // (2,1)
}

fn test_identity_single<T>()
where
    T: Num + Clone + Copy + PartialEq + Debug + Zero + One,
{
    let i = Matrix::<T>::identity(1).unwrap();
    assert_eq!(i.rows(), 1);
    assert_eq!(i.cols(), 1);
    assert_eq!(i.as_slice()[0], T::one());
}

// -------- f64 Tests --------

#[test]
fn test_matrix_new_valid_f64() { test_new_valid::<f64>(); }

#[test]
fn test_matrix_new_dimension_mismatch_f64() { test_new_dimension_mismatch::<f64>(); }

#[test]
fn test_matrix_full_f64() { test_full::<f64>(); }

#[test]
fn test_matrix_zeroes_f64() { test_zeroes::<f64>(); }

#[test]
fn test_matrix_ones_f64() { test_ones::<f64>(); }

#[test]
fn test_matrix_as_slice_f64() { test_as_slice::<f64>(); }

#[test]
fn test_matrix_as_mut_slice_f64() { test_as_mut_slice::<f64>(); }

#[test]
fn test_matrix_single_element_f64() { test_single_element::<f64>(); }

#[test]
fn test_matrix_single_row_f64() { test_single_row::<f64>(); }

#[test]
fn test_matrix_single_column_f64() { test_single_column::<f64>(); }

#[test]
fn test_matrix_identity_f64() { test_identity::<f64>(); }

#[test]
fn test_matrix_identity_single_f64() { test_identity_single::<f64>(); }

// -------- f32 Tests --------

#[test]
fn test_matrix_new_valid_f32() { test_new_valid::<f32>(); }

#[test]
fn test_matrix_new_dimension_mismatch_f32() { test_new_dimension_mismatch::<f32>(); }

#[test]
fn test_matrix_full_f32() { test_full::<f32>(); }

#[test]
fn test_matrix_zeroes_f32() { test_zeroes::<f32>(); }

#[test]
fn test_matrix_ones_f32() { test_ones::<f32>(); }

#[test]
fn test_matrix_as_slice_f32() { test_as_slice::<f32>(); }

#[test]
fn test_matrix_as_mut_slice_f32() { test_as_mut_slice::<f32>(); }

#[test]
fn test_matrix_single_element_f32() { test_single_element::<f32>(); }

#[test]
fn test_matrix_identity_f32() { test_identity::<f32>(); }

#[test]
fn test_matrix_identity_single_f32() { test_identity_single::<f32>(); }

// -------- i32 Tests --------

#[test]
fn test_matrix_new_valid_i32() { test_new_valid::<i32>(); }

#[test]
fn test_matrix_new_dimension_mismatch_i32() { test_new_dimension_mismatch::<i32>(); }

#[test]
fn test_matrix_full_i32() { test_full::<i32>(); }

#[test]
fn test_matrix_zeroes_i32() { test_zeroes::<i32>(); }

#[test]
fn test_matrix_ones_i32() { test_ones::<i32>(); }

#[test]
fn test_matrix_as_slice_i32() { test_as_slice::<i32>(); }

#[test]
fn test_matrix_as_mut_slice_i32() { test_as_mut_slice::<i32>(); }

#[test]
fn test_matrix_single_element_i32() { test_single_element::<i32>(); }

#[test]
fn test_matrix_identity_i32() { test_identity::<i32>(); }

#[test]
fn test_matrix_identity_single_i32() { test_identity_single::<i32>(); }

// -------- i64 Tests --------

#[test]
fn test_matrix_new_valid_i64() { test_new_valid::<i64>(); }

#[test]
fn test_matrix_zeroes_i64() { test_zeroes::<i64>(); }

#[test]
fn test_matrix_ones_i64() { test_ones::<i64>(); }

#[test]
fn test_matrix_identity_i64() { test_identity::<i64>(); }

// -------- Float-specific Edge Cases --------

#[test]
fn test_matrix_new_empty_vector() {
    let m = Matrix::<f64>::new(2, 2, vec![]);
    assert!(m.is_err());
}

#[test]
fn test_matrix_new_with_infinity() {
    let m = Matrix::new(2, 2, vec![f64::INFINITY, f64::NEG_INFINITY, 1.0, 2.0]).unwrap();
    assert_eq!(m.as_slice()[0], f64::INFINITY);
    assert_eq!(m.as_slice()[1], f64::NEG_INFINITY);
}

#[test]
fn test_matrix_new_with_nan() {
    let m = Matrix::new(2, 2, vec![f64::NAN, 1.0, 2.0, 3.0]).unwrap();
    assert!(m.as_slice()[0].is_nan());
}

#[test]
fn test_matrix_display() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let display = format!("{}", m);
    assert!(display.contains("1"));
    assert!(display.contains("4"));
}

#[test]
fn test_matrix_identity_zero_dimension() {
    let result = Matrix::<f64>::identity(0);
    assert!(result.is_err());
}
