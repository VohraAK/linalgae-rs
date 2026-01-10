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

// -------- Apply Tests --------

#[test]
fn test_apply_square() {
    let m = Matrix::new(2, 2, vec![1.0, 4.0, 9.0, 16.0]).unwrap();
    
    let result = m.apply(|x| x.sqrt());
    
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_apply_double() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    let result = m.apply(|x| x * 2.0);
    
    assert_eq!(result.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_apply_negate() {
    let m = Matrix::new(2, 2, vec![1.0, -2.0, 3.0, -4.0]).unwrap();
    
    let result = m.apply(|x| -x);
    
    assert_eq!(result.as_slice(), &[-1.0, 2.0, -3.0, 4.0]);
}

#[test]
fn test_apply_abs() {
    let m = Matrix::new(2, 2, vec![-1.0, 2.0, -3.0, 4.0]).unwrap();
    
    let result = m.apply(|x| x.abs());
    
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_apply_closure_with_capture() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let offset = 10.0;
    
    let result = m.apply(|x| x + offset);
    
    assert_eq!(result.as_slice(), &[11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn test_apply_preserves_dimensions() {
    let m = Matrix::new(3, 4, vec![1.0; 12]).unwrap();
    
    let result = m.apply(|x| x * 5.0);
    
    assert_eq!(result.rows(), 3);
    assert_eq!(result.cols(), 4);
}

#[test]
fn test_apply_chained() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    // square then add 1
    let result = m.apply(|x| x * x).apply(|x| x + 1.0);
    
    assert_eq!(result.as_slice(), &[2.0, 5.0, 10.0, 17.0]);
}

// -------- Matrix Multiplication Edge Cases --------

#[test]
fn test_matrix_mul_single_element() {
    let a = Matrix::new(1, 1, vec![3.0]).unwrap();
    let b = Matrix::new(1, 1, vec![4.0]).unwrap();
    
    let c = &a * &b;
    
    assert_eq!(c.as_slice(), &[12.0]);
}

#[test]
fn test_matrix_mul_with_zeros() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let zero = Matrix::zeroes(2, 2).unwrap();
    
    let c = &a * &zero;
    
    for val in c.as_slice() {
        assert_eq!(*val, 0.0);
    }
}

#[test]
fn test_matrix_mul_row_vector_times_col_vector() {
    // 1x3 * 3x1 = 1x1 (dot product)
    let row = Matrix::new(1, 3, vec![1.0, 2.0, 3.0]).unwrap();
    let col = Matrix::new(3, 1, vec![4.0, 5.0, 6.0]).unwrap();
    
    let result = &row * &col;
    
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert_eq!(result.rows(), 1);
    assert_eq!(result.cols(), 1);
    assert_eq!(result.as_slice(), &[32.0]);
}

#[test]
fn test_matrix_mul_col_vector_times_row_vector() {
    // 3x1 * 1x3 = 3x3 (outer product)
    let col = Matrix::new(3, 1, vec![1.0, 2.0, 3.0]).unwrap();
    let row = Matrix::new(1, 3, vec![4.0, 5.0, 6.0]).unwrap();
    
    let result = &col * &row;
    
    assert_eq!(result.rows(), 3);
    assert_eq!(result.cols(), 3);
    assert_eq!(result.as_slice(), &[4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0]);
}

#[test]
fn test_matrix_mul_negative_values() {
    let a = Matrix::new(2, 2, vec![-1.0, 2.0, -3.0, 4.0]).unwrap();
    let b = Matrix::new(2, 2, vec![5.0, -6.0, 7.0, -8.0]).unwrap();
    
    let c = &a * &b;
    
    // [-1,2] * [5, -6]   [-1*5+2*7, -1*-6+2*-8]   [9, -10]
    // [-3,4]   [7, -8] = [-3*5+4*7, -3*-6+4*-8] = [13, -14]
    assert_eq!(c.as_slice(), &[9.0, -10.0, 13.0, -14.0]);
}

#[test]
#[should_panic(expected = "lhs_cols")]
fn test_matrix_mul_dimension_mismatch_reverse() {
    let a = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    
    let _ = &a * &b; // 3x2 * 3x2 should fail (need 2x3 for second)
}

// -------- Transpose Edge Cases --------

#[test]
fn test_transpose_with_negative_values() {
    let m = Matrix::new(2, 2, vec![-1.0, -2.0, -3.0, -4.0]).unwrap();
    
    let t = m.t();
    
    assert_eq!(t.as_slice(), &[-1.0, -3.0, -2.0, -4.0]);
}

#[test]
fn test_transpose_preserves_values() {
    let m = Matrix::new(2, 3, vec![1.5, 2.7, 3.9, 4.2, 5.8, 6.1]).unwrap();
    
    let t = m.t();
    
    // verify all original values exist in transpose
    assert!(t.as_slice().contains(&1.5));
    assert!(t.as_slice().contains(&2.7));
    assert!(t.as_slice().contains(&6.1));
}

#[test]
fn test_transpose_large_non_square() {
    let data: Vec<f64> = (0..60).map(|i| i as f64).collect();
    let m = Matrix::new(10, 6, data).unwrap();
    
    let t = m.t();
    
    assert_eq!(t.rows(), 6);
    assert_eq!(t.cols(), 10);
}

// -------- Apply Edge Cases --------

#[test]
fn test_apply_identity() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    let result = m.apply(|x| x);
    
    assert_eq!(result.as_slice(), m.as_slice());
}

#[test]
fn test_apply_constant() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    let result = m.apply(|_| 42.0);
    
    for val in result.as_slice() {
        assert_eq!(*val, 42.0);
    }
}

#[test]
fn test_apply_single_element() {
    let m = Matrix::new(1, 1, vec![5.0]).unwrap();
    
    let result = m.apply(|x| x * 3.0);
    
    assert_eq!(result.as_slice(), &[15.0]);
}

#[test]
fn test_apply_with_zero() {
    let m = Matrix::zeroes(3, 3).unwrap();
    
    let result = m.apply(|x| x + 1.0);
    
    for val in result.as_slice() {
        assert_eq!(*val, 1.0);
    }
}

#[test]
fn test_apply_floor() {
    let m = Matrix::new(2, 2, vec![1.7, 2.3, 3.9, 4.1]).unwrap();
    
    let result = m.apply(|x| x.floor());
    
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_apply_ceil() {
    let m = Matrix::new(2, 2, vec![1.1, 2.2, 3.3, 4.4]).unwrap();
    
    let result = m.apply(|x| x.ceil());
    
    assert_eq!(result.as_slice(), &[2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_apply_complex_function() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    // Apply a more complex function: f(x) = 2x^2 + 3x + 1
    let result = m.apply(|x| 2.0 * x * x + 3.0 * x + 1.0);
    
    assert_eq!(result.as_slice(), &[6.0, 15.0, 28.0, 45.0]);
}

#[test]
fn test_apply_with_nan_handling() {
    let m = Matrix::new(2, 2, vec![1.0, -1.0, 4.0, -4.0]).unwrap();
    
    // sqrt of negative numbers produces NaN
    let result = m.apply(|x| if x >= 0.0 { x.sqrt() } else { 0.0 });
    
    assert_eq!(result.as_slice(), &[1.0, 0.0, 2.0, 0.0]);
}

// -------- Combined Operations Edge Cases --------

#[test]
fn test_transpose_then_multiply() {
    let a = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Matrix::new(2, 3, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
    
    // A^T * B should work (3x2 * 2x3 = 3x3)
    let result = &a.t() * &b;
    
    assert_eq!(result.rows(), 3);
    assert_eq!(result.cols(), 3);
}

#[test]
fn test_apply_then_transpose() {
    let m = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    
    let result = m.apply(|x| x * 2.0).t();
    
    assert_eq!(result.rows(), 3);
    assert_eq!(result.cols(), 2);
    assert_eq!(result.as_slice()[0], 2.0); // first element should be 2.0
    assert_eq!(result.as_slice()[1], 8.0); // should be 4.0 * 2
}
