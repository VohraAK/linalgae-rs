use std::ops;
use crate::core::matrix::Matrix;


// ----------Arithmetic Operations----------//

// addition
impl ops::Add<&Matrix> for &Matrix
{
    type Output = Matrix;

    fn add(self, rhs: &Matrix) -> Self::Output 
    {
        let lhs_rows = self.rows();
        let lhs_cols = self.cols();
        let rhs_rows = rhs.rows();
        let rhs_cols = rhs.cols();

        // ensure that the dims match up
        if lhs_rows != rhs_rows || lhs_cols != rhs_cols
        {
            panic!("Matrix::Add: Dimension mismatch! ( ({}, {}) vs ({}, {}) )", lhs_rows, lhs_cols, rhs_rows, rhs_cols);
        }

        let result = self.as_slice().iter().zip(rhs.as_slice().iter()).map(|(a, b)| a + b ).collect();

        Matrix::new(lhs_rows, lhs_cols, result).expect("Matrix::Add: Result has incorrect dimensions!")
        
    }
}

// subtraction
impl ops::Sub<&Matrix> for &Matrix
{
    type Output = Matrix;

    fn sub(self, rhs: &Matrix) -> Self::Output 
    {
        let lhs_rows = self.rows();
        let lhs_cols = self.cols();
        let rhs_rows = rhs.rows();
        let rhs_cols = rhs.cols();

        // ensure that the dims match up
        if lhs_rows != rhs_rows || lhs_cols != rhs_cols
        {
            panic!("Matrix::Sub: Dimension mismatch! ( ({}, {}) vs ({}, {}) )", lhs_rows, lhs_cols, rhs_rows, rhs_cols);
        }

        let result = self.as_slice().iter().zip(rhs.as_slice().iter()).map(|(a, b)| a - b ).collect();

        Matrix::new(lhs_rows, lhs_cols, result).expect("Matrix::Sub: Result has incorrect dimensions!")
        
    }
}

// scalar mul (Matrix * scalar)
impl ops::Mul<f64> for &Matrix
{
    type Output = Matrix;

    fn mul(self, rhs: f64) -> Self::Output 
    {
        let lhs_rows = self.rows();
        let lhs_cols = self.cols();

        let result = self.as_slice().iter().map(|a|  a * rhs).collect();

        Matrix::new(lhs_rows, lhs_cols, result).expect("Matrix::Mul: Result has incorrect dimensions!")
        
    }
}

// scalar mul (scalar * Matrix)
impl ops::Mul<&Matrix> for f64
{
    type Output = Matrix;

    fn mul(self, rhs: &Matrix) -> Self::Output { rhs * self }
}

// scalar div (Matrix / scalar)
impl ops::Div<f64> for &Matrix
{
    type Output = Matrix;

    fn div(self, rhs: f64) -> Self::Output 
    {
        // TODO: epsilon
        // check 0-divisor
        if rhs == (0 as f64)
        {
            panic!("Matrix::Div: Cannot divide by zero!");
        }

        let lhs_rows = self.rows();
        let lhs_cols = self.cols();

        let result = self.as_slice().iter().map(|a| a / rhs).collect();

        Matrix::new(lhs_rows, lhs_cols, result).expect("Matrix::Div: Result has incorrect dimensions!") 
    }
}