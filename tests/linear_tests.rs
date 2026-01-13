use linalgae_rs::{matrix, core::matrix::Matrix};
use num_traits::{Zero, One, Num};
use std::fmt::Debug;
use std::ops::Mul;

// -------- Generic Matmul Helpers --------

fn test_matmul_basic<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + From<u8> + std::ops::AddAssign,
    for<'a> &'a Matrix<T>: Mul<&'a Matrix<T>, Output = Matrix<T>>,
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
    let c = &a * &b;
    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 2);
    assert_eq!(c.as_slice(), &[T::from(58), T::from(64), T::from(139), T::from(154)]);
}

fn test_matmul_identity<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + Zero + One + From<u8> + std::ops::AddAssign,
    for<'a> &'a Matrix<T>: Mul<&'a Matrix<T>, Output = Matrix<T>>,
{
    let a = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    // Identity matrix
    let i = matrix![[T::one(), T::zero()], [T::zero(), T::one()]];
    let c = &a * &i;
    assert_eq!(c.as_slice(), a.as_slice());
}

fn test_matmul_zeros<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + Zero + From<u8> + std::ops::AddAssign,
    for<'a> &'a Matrix<T>: Mul<&'a Matrix<T>, Output = Matrix<T>>,
{
    let a = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    let z = Matrix::<T>::zeroes(2, 2).unwrap();
    let c = &a * &z;
    for val in c.as_slice() {
        assert_eq!(*val, T::zero());
    }
}

fn test_matmul_single_element<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + From<u8> + std::ops::AddAssign,
    for<'a> &'a Matrix<T>: Mul<&'a Matrix<T>, Output = Matrix<T>>,
{
    let a = matrix![[T::from(3)]];
    let b = matrix![[T::from(4)]];
    let c = &a * &b;
    assert_eq!(c.as_slice()[0], T::from(12));
}

fn test_matmul_row_vector_column_vector<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + From<u8> + std::ops::AddAssign,
    for<'a> &'a Matrix<T>: Mul<&'a Matrix<T>, Output = Matrix<T>>,
{
    // Row vector (1x3) * column vector (3x1) = scalar (1x1)
    let row = matrix![[T::from(1), T::from(2), T::from(3)]];
    let col = matrix![[T::from(4)], [T::from(5)], [T::from(6)]];
    let c = &row * &col;
    assert_eq!(c.rows(), 1);
    assert_eq!(c.cols(), 1);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_eq!(c.as_slice()[0], T::from(32));
}

fn test_matmul_column_vector_row_vector<T>()
where
    T: Num + Copy + Clone + PartialEq + Debug + Default + From<u8> + std::ops::AddAssign,
    for<'a> &'a Matrix<T>: Mul<&'a Matrix<T>, Output = Matrix<T>>,
{
    // Column vector (3x1) * row vector (1x3) = outer product (3x3)
    let col = matrix![[T::from(1)], [T::from(2)], [T::from(3)]];
    let row = matrix![[T::from(4), T::from(5), T::from(6)]];
    let c = &col * &row;
    assert_eq!(c.rows(), 3);
    assert_eq!(c.cols(), 3);
    assert_eq!(c.as_slice(), &[
        T::from(4), T::from(5), T::from(6),
        T::from(8), T::from(10), T::from(12),
        T::from(12), T::from(15), T::from(18),
    ]);
}

// -------- Generic Transpose Helpers --------

fn test_transpose_basic<T>()
where
    T: Num + Clone + Copy + PartialEq + Debug + Default + From<u8>,
{
    let m = matrix![
        [T::from(1), T::from(2), T::from(3)],
        [T::from(4), T::from(5), T::from(6)]
    ];
    let t = m.t();
    assert_eq!(t.rows(), 3);
    assert_eq!(t.cols(), 2);
    assert_eq!(t.as_slice(), &[
        T::from(1), T::from(4),
        T::from(2), T::from(5),
        T::from(3), T::from(6),
    ]);
}

fn test_transpose_square<T>()
where
    T: Num + Clone + Copy + PartialEq + Debug + Default + From<u8>,
{
    let m = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    let t = m.t();
    assert_eq!(t.as_slice(), &[T::from(1), T::from(3), T::from(2), T::from(4)]);
}

fn test_transpose_row_vector<T>()
where
    T: Num + Clone + Copy + PartialEq + Debug + Default + From<u8>,
{
    let m = Matrix::new(1, 4, vec![T::from(1), T::from(2), T::from(3), T::from(4)]).unwrap();
    let t = m.t();
    assert_eq!(t.rows(), 4);
    assert_eq!(t.cols(), 1);
}

fn test_transpose_column_vector<T>()
where
    T: Num + Clone + Copy + PartialEq + Debug + Default + From<u8>,
{
    let m = Matrix::new(4, 1, vec![T::from(1), T::from(2), T::from(3), T::from(4)]).unwrap();
    let t = m.t();
    assert_eq!(t.rows(), 1);
    assert_eq!(t.cols(), 4);
}

fn test_transpose_single_element<T>()
where
    T: Num + Clone + Copy + PartialEq + Debug + Default + From<u8>,
{
    let m = matrix![[T::from(42)]];
    let t = m.t();
    assert_eq!(t.rows(), 1);
    assert_eq!(t.cols(), 1);
    assert_eq!(t.as_slice()[0], T::from(42));
}

fn test_double_transpose<T>()
where
    T: Num + Clone + Copy + PartialEq + Debug + Default + From<u8>,
{
    let m = matrix![
        [T::from(1), T::from(2), T::from(3)],
        [T::from(4), T::from(5), T::from(6)]
    ];
    let tt = m.t().t();
    assert_eq!(tt.as_slice(), m.as_slice());
}

// -------- Generic Apply Helpers --------

fn test_apply_double<T>()
where
    T: Num + Clone + Copy + PartialEq + Debug + From<u8>,
    T: std::ops::Add<Output = T>,
{
    let m = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    let doubled = m.apply(|x| x + x);
    assert_eq!(doubled.as_slice(), &[T::from(2), T::from(4), T::from(6), T::from(8)]);
}

fn test_apply_identity<T>()
where
    T: Num + Clone + Copy + PartialEq + Debug + From<u8>,
{
    let m = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    let same = m.apply(|x| x);
    assert_eq!(same.as_slice(), m.as_slice());
}

fn test_apply_to_constant<T>()
where
    T: Num + Clone + Copy + PartialEq + Debug + From<u8>,
{
    let m = matrix![[T::from(1), T::from(2)], [T::from(3), T::from(4)]];
    let constant = m.apply(|_| T::from(99));
    for val in constant.as_slice() {
        assert_eq!(*val, T::from(99));
    }
}

fn test_apply_single_element<T>()
where
    T: Num + Clone + Copy + PartialEq + Debug + From<u8>,
{
    let m = matrix![[T::from(5)]];
    let applied = m.apply(|x| x * T::from(10));
    assert_eq!(applied.as_slice()[0], T::from(50));
}

// -------- f64 Matmul Tests --------

#[test]
fn test_matmul_basic_f64() { test_matmul_basic::<f64>(); }

#[test]
fn test_matmul_identity_f64() { test_matmul_identity::<f64>(); }

#[test]
fn test_matmul_zeros_f64() { test_matmul_zeros::<f64>(); }

#[test]
fn test_matmul_single_element_f64() { test_matmul_single_element::<f64>(); }

#[test]
fn test_matmul_row_vector_column_vector_f64() { test_matmul_row_vector_column_vector::<f64>(); }

#[test]
fn test_matmul_column_vector_row_vector_f64() { test_matmul_column_vector_row_vector::<f64>(); }

// -------- f32 Matmul Tests --------

#[test]
fn test_matmul_basic_f32() { test_matmul_basic::<f32>(); }

#[test]
fn test_matmul_identity_f32() { test_matmul_identity::<f32>(); }

#[test]
fn test_matmul_zeros_f32() { test_matmul_zeros::<f32>(); }

// -------- i32 Matmul Tests --------

#[test]
fn test_matmul_basic_i32() { test_matmul_basic::<i32>(); }

#[test]
fn test_matmul_identity_i32() { test_matmul_identity::<i32>(); }

#[test]
fn test_matmul_zeros_i32() { test_matmul_zeros::<i32>(); }

#[test]
fn test_matmul_single_element_i32() { test_matmul_single_element::<i32>(); }

// -------- i64 Matmul Tests --------

#[test]
fn test_matmul_basic_i64() { test_matmul_basic::<i64>(); }

// -------- f64 Transpose Tests --------

#[test]
fn test_transpose_basic_f64() { test_transpose_basic::<f64>(); }

#[test]
fn test_transpose_square_f64() { test_transpose_square::<f64>(); }

#[test]
fn test_transpose_row_vector_f64() { test_transpose_row_vector::<f64>(); }

#[test]
fn test_transpose_column_vector_f64() { test_transpose_column_vector::<f64>(); }

#[test]
fn test_transpose_single_element_f64() { test_transpose_single_element::<f64>(); }

#[test]
fn test_double_transpose_f64() { test_double_transpose::<f64>(); }

// -------- f32 Transpose Tests --------

#[test]
fn test_transpose_basic_f32() { test_transpose_basic::<f32>(); }

#[test]
fn test_transpose_square_f32() { test_transpose_square::<f32>(); }

#[test]
fn test_double_transpose_f32() { test_double_transpose::<f32>(); }

// -------- i32 Transpose Tests --------

#[test]
fn test_transpose_basic_i32() { test_transpose_basic::<i32>(); }

#[test]
fn test_transpose_square_i32() { test_transpose_square::<i32>(); }

#[test]
fn test_double_transpose_i32() { test_double_transpose::<i32>(); }

// -------- f64 Apply Tests --------

#[test]
fn test_apply_double_f64() { test_apply_double::<f64>(); }

#[test]
fn test_apply_identity_f64() { test_apply_identity::<f64>(); }

#[test]
fn test_apply_to_constant_f64() { test_apply_to_constant::<f64>(); }

#[test]
fn test_apply_single_element_f64() { test_apply_single_element::<f64>(); }

// -------- f32 Apply Tests --------

#[test]
fn test_apply_double_f32() { test_apply_double::<f32>(); }

#[test]
fn test_apply_identity_f32() { test_apply_identity::<f32>(); }

// -------- i32 Apply Tests --------

#[test]
fn test_apply_double_i32() { test_apply_double::<i32>(); }

#[test]
fn test_apply_identity_i32() { test_apply_identity::<i32>(); }

#[test]
fn test_apply_to_constant_i32() { test_apply_to_constant::<i32>(); }

// -------- Float-specific Tests --------

#[test]
fn test_apply_sqrt() {
    let m = matrix![[4.0_f64, 9.0], [16.0, 25.0]];
    let sqrts = m.apply(|x| x.sqrt());
    assert_eq!(sqrts.as_slice(), &[2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_apply_floor() {
    let m = matrix![[1.1_f64, 2.9], [3.5, 4.0]];
    let floors = m.apply(|x| x.floor());
    assert_eq!(floors.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_apply_ceil() {
    let m = matrix![[1.1_f64, 2.9], [3.5, 4.0]];
    let ceils = m.apply(|x| x.ceil());
    assert_eq!(ceils.as_slice(), &[2.0, 3.0, 4.0, 4.0]);
}

#[test]
#[should_panic(expected = "lhs_cols")]
fn test_matmul_dimension_mismatch() {
    let a = matrix![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let b = matrix![[1.0, 2.0], [3.0, 4.0]];
    let _ = &a * &b;
}

#[test]
fn test_matmul_with_infinity() {
    let a = matrix![[f64::INFINITY, 1.0], [2.0, 3.0]];
    let b = matrix![[1.0, 2.0], [3.0, 4.0]];
    let c = &a * &b;
    assert_eq!(c.as_slice()[0], f64::INFINITY);
}

#[test]
fn test_transpose_with_nan() {
    let m = matrix![[f64::NAN, 1.0], [2.0, 3.0]];
    let t = m.t();
    assert!(t.as_slice()[0].is_nan());
}

#[test]
fn test_apply_with_nan() {
    let m = matrix![[f64::NAN]];
    let applied = m.apply(|x| x + 1.0);
    assert!(applied.as_slice()[0].is_nan());
}
