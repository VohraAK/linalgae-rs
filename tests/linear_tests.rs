use linalgae_rs::core::matrix::Matrix;

// -------- Matrix Multiplication Tests --------

#[test]
fn test_matrix_mul_square() {
    // 2x2 * 2x2
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    
    let c = &a * &b;
    
    // [1,2] * [5,6] = [1*5+2*7, 1*6+2*8] = [19, 22]
    // [3,4]   [7,8]   [3*5+4*7, 3*6+4*8]   [43, 50]
    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 2);
    assert_eq!(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_matrix_mul_non_square() {
    // 2x3 * 3x2 = 2x2
    let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Matrix::new(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    
    let c = &a * &b;
    
    // [1,2,3] * [7, 8 ]   [1*7+2*9+3*11,  1*8+2*10+3*12]   [58,  64]
    // [4,5,6]   [9, 10]   [4*7+5*9+6*11,  4*8+5*10+6*12]   [139, 154]
    //           [11,12]
    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 2);
    assert_eq!(c.as_slice(), &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn test_matrix_mul_identity() {
    // multiplying by identity should return the same matrix
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let identity = Matrix::new(2, 2, vec![1.0, 0.0, 0.0, 1.0]).unwrap();
    
    let c = &a * &identity;
    
    assert_eq!(c.as_slice(), a.as_slice());
}

#[test]
fn test_matrix_mul_vector() {
    // 2x3 * 3x1 = 2x1 (matrix-vector multiplication)
    let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let v = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
    
    let c = &a * &v;
    
    // [1,2,3] * [1]   [1*1+2*2+3*3]   [14]
    // [4,5,6]   [2] = [4*1+5*2+6*3] = [32]
    //           [3]
    assert_eq!(c.rows(), 2);
    assert_eq!(c.cols(), 1);
    assert_eq!(c.as_slice(), &[14.0, 32.0]);
}

#[test]
#[should_panic(expected = "lhs_cols")]
fn test_matrix_mul_dimension_mismatch() {
    // 2x2 * 3x2 - incompatible dimensions
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    
    let _ = &a * &b; // should panic
}

// -------- Transpose Tests --------

#[test]
fn test_transpose_square() {
    // 2x2 matrix
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    let t = m.t();
    
    // [1, 2]^T = [1, 3]
    // [3, 4]     [2, 4]
    assert_eq!(t.rows(), 2);
    assert_eq!(t.cols(), 2);
    assert_eq!(t.as_slice(), &[1.0, 3.0, 2.0, 4.0]);
}

#[test]
fn test_transpose_non_square() {
    // 2x3 matrix
    let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    
    let t = m.t();
    
    // [1, 2, 3]^T = [1, 4]
    // [4, 5, 6]     [2, 5]
    //               [3, 6]
    assert_eq!(t.rows(), 3);
    assert_eq!(t.cols(), 2);
    assert_eq!(t.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn test_transpose_row_vector() {
    // 1x4 row vector becomes 4x1 column vector
    let m = Matrix::new(1, 4, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    let t = m.t();
    
    assert_eq!(t.rows(), 4);
    assert_eq!(t.cols(), 1);
    assert_eq!(t.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_transpose_column_vector() {
    // 4x1 column vector becomes 1x4 row vector
    let m = Matrix::new(4, 1, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    let t = m.t();
    
    assert_eq!(t.rows(), 1);
    assert_eq!(t.cols(), 4);
    assert_eq!(t.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_transpose_double() {
    // (A^T)^T = A
    let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    
    let t_t = m.t().t();
    
    assert_eq!(t_t.rows(), m.rows());
    assert_eq!(t_t.cols(), m.cols());
    assert_eq!(t_t.as_slice(), m.as_slice());
}

#[test]
fn test_transpose_single_element() {
    // 1x1 matrix - transpose is itself
    let m = Matrix::new(1, 1, vec![42.0]).unwrap();
    
    let t = m.t();
    
    assert_eq!(t.rows(), 1);
    assert_eq!(t.cols(), 1);
    assert_eq!(t.as_slice(), &[42.0]);
}
