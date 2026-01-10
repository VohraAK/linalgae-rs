use linalgae_rs::core::matrix::Matrix;

// -------- Addition Tests --------

#[test]
fn test_matrix_addition() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    
    let c = &a + &b;
    
    assert_eq!(c.as_slice(), &[6.0, 8.0, 10.0, 12.0]);
}

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_matrix_addition_dimension_mismatch() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    
    let _ = &a + &b; // should panic
}

// -------- Subtraction Tests --------

#[test]
fn test_matrix_subtraction() {
    let a = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let b = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    let c = &a - &b;
    
    assert_eq!(c.as_slice(), &[4.0, 4.0, 4.0, 4.0]);
}

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_matrix_subtraction_dimension_mismatch() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    
    let _ = &a - &b; // should panic
}

// -------- Scalar Multiplication Tests --------

#[test]
fn test_matrix_scalar_mul() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    let result = &m * 2.0;
    
    assert_eq!(result.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_scalar_matrix_mul() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    let result = 3.0 * &m;
    
    assert_eq!(result.as_slice(), &[3.0, 6.0, 9.0, 12.0]);
}

#[test]
fn test_matrix_scalar_mul_zero() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    let result = &m * 0.0;
    
    for val in result.as_slice() {
        assert_eq!(*val, 0.0);
    }
}

// -------- Scalar Division Tests --------

#[test]
fn test_matrix_scalar_div() {
    let m = Matrix::new(2, 2, vec![2.0, 4.0, 6.0, 8.0]).unwrap();
    
    let result = &m / 2.0;
    
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
#[should_panic(expected = "Cannot divide by zero")]
fn test_matrix_scalar_div_by_zero() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    let _ = &m / 0.0; // should panic
}

// -------- Edge Case Tests --------

#[test]
fn test_matrix_addition_single_element() {
    let a = Matrix::new(1, 1, vec![5.0]).unwrap();
    let b = Matrix::new(1, 1, vec![3.0]).unwrap();
    
    let c = &a + &b;
    
    assert_eq!(c.as_slice(), &[8.0]);
}

#[test]
fn test_matrix_addition_negative_values() {
    let a = Matrix::new(2, 2, vec![-1.0, -2.0, -3.0, -4.0]).unwrap();
    let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    
    let c = &a + &b;
    
    assert_eq!(c.as_slice(), &[4.0, 4.0, 4.0, 4.0]);
}

#[test]
fn test_matrix_addition_with_zero_matrix() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let zero = Matrix::zeroes(2, 2).unwrap();
    
    let c = &a + &zero;
    
    assert_eq!(c.as_slice(), a.as_slice());
}

#[test]
fn test_matrix_subtraction_result_negative() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    
    let c = &a - &b;
    
    assert_eq!(c.as_slice(), &[-4.0, -4.0, -4.0, -4.0]);
}

#[test]
fn test_matrix_subtraction_self() {
    let a = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    let c = &a - &a;
    
    for val in c.as_slice() {
        assert_eq!(*val, 0.0);
    }
}

#[test]
fn test_matrix_scalar_mul_negative() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    let result = &m * -2.0;
    
    assert_eq!(result.as_slice(), &[-2.0, -4.0, -6.0, -8.0]);
}

#[test]
fn test_matrix_scalar_mul_one() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    let result = &m * 1.0;
    
    assert_eq!(result.as_slice(), m.as_slice());
}

#[test]
fn test_matrix_scalar_mul_fractional() {
    let m = Matrix::new(2, 2, vec![2.0, 4.0, 6.0, 8.0]).unwrap();
    
    let result = &m * 0.5;
    
    assert_eq!(result.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_matrix_scalar_div_one() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    let result = &m / 1.0;
    
    assert_eq!(result.as_slice(), m.as_slice());
}

#[test]
fn test_matrix_scalar_div_negative() {
    let m = Matrix::new(2, 2, vec![2.0, 4.0, 6.0, 8.0]).unwrap();
    
    let result = &m / -2.0;
    
    assert_eq!(result.as_slice(), &[-1.0, -2.0, -3.0, -4.0]);
}

#[test]
fn test_matrix_scalar_div_fractional() {
    let m = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    let result = &m / 0.5;
    
    assert_eq!(result.as_slice(), &[2.0, 4.0, 6.0, 8.0]);
}

#[test]
#[should_panic(expected = "Dimension mismatch")]
fn test_matrix_addition_row_col_mismatch() {
    let a = Matrix::new(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Matrix::new(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    
    let _ = &a + &b;
}

#[test]
fn test_matrix_scalar_operations_large() {
    let m = Matrix::ones(100, 100).unwrap();
    
    let result = &(&m * 2.0) / 2.0;
    
    for val in result.as_slice() {
        assert_eq!(*val, 1.0);
    }
}
