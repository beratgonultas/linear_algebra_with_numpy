# Linear Algebra with NumPy

NumPy is a nice tool and here I introduced some numpy functions about linear algebra.

- numpy.linalg.det(a)
- numpy.linalg.eig(a)
- numpy.linalg.eigvals(a)
- numpy.linalg.inv(a)
- numpy.linalg.solve(a, b)


Let's begin by importing Numpy and listing out the functions covered in this notebook.


```python
import numpy as np
```


```python
# List of functions explained 
function1 = np.linalg.det  # Computes determinant
function2 = np.linalg.eig  # Computes eigenvectors and eigenvalues
function3 = np.linalg.eigvals # Computes only eigenvalues
function4 = np.linalg.inv # Computes the inverse 
function5 = np.linalg.solve # Solves the matrix equation
```

## Function 1 - numpy.linalg.det(a)

This function computes the determinant of an array.


```python
# Example 1 - Determinant of 2D array
arr1 = [[1, 2], 
        [3, 4.]]

np.linalg.det(arr1)
```




    -2.0000000000000004



Above, you see a basic example consisting a 2D array. As you can see, the function gave us the determinant of the array. 


```python
# Example 2 - Determinant of an array of shape (3, 2, 2)
arr2 = [[[1, 2],
        [3, 4]]
        ,
        
        [[5, 6],
        [7, 8]]
        ,
        
        [[9, 10],
        [11, 12]]]
np.linalg.det(arr2)
```




    array([-2., -2., -2.])



Above we see the result for an array consisting 3 matrix, that is, an array with shape (3, 2, 2). As you can see, it computes the determinants of matrices separately.


```python
# Example 3 - Let's try a 2x3 array
arr3 = [[1, 2, 4], 
        [3, 4, 6]]


np.linalg.det(arr3)
```


    ---------------------------------------------------------------------------

    LinAlgError                               Traceback (most recent call last)

    /tmp/ipykernel_37/3491642392.py in <module>
          4 
          5 
    ----> 6 np.linalg.det(arr3)
    

    <__array_function__ internals> in det(*args, **kwargs)


    /opt/conda/lib/python3.9/site-packages/numpy/linalg/linalg.py in det(a)
       2153     a = asarray(a)
       2154     _assert_stacked_2d(a)
    -> 2155     _assert_stacked_square(a)
       2156     t, result_t = _commonType(a)
       2157     signature = 'D->D' if isComplexType(t) else 'd->d'


    /opt/conda/lib/python3.9/site-packages/numpy/linalg/linalg.py in _assert_stacked_square(*arrays)
        201         m, n = a.shape[-2:]
        202         if m != n:
    --> 203             raise LinAlgError('Last 2 dimensions of the array must be square')
        204 
        205 def _assert_finite(*arrays):


    LinAlgError: Last 2 dimensions of the array must be square


Above example breaks, because it does not consist of square matrices. Its shape is (2, 3). As it is stated in the error message: "Last 2 dimensions of the array must be square"

This is a nice and useful function. It might be very helpful for calculating determinants of big arrays like (50, 50) 

## Function 2 - numpy.linalg.eig(a)

This function evaluate the eigenvalues and eigenvectors of an array.


```python
# Let's remember what was arr1
arr1
```




    [[1, 2], [3, 4.0]]




```python
# Example 1 - Calculation of eigenvalues and eigenvectors of arr1
np.linalg.eig(arr1)
```




    (array([-0.37228132,  5.37228132]),
     array([[-0.82456484, -0.41597356],
            [ 0.56576746, -0.90937671]]))



Above you see two arrays as outputs. The first one is the array of eigenvalues. The second one is the array of eigenvectors.


```python
# Let's define arr4, an 3x3 matrix
arr4 = [[1, 2, 4], [3, 4, 6], [7, 8, 9]]
arr4
```




    [[1, 2, 4], [3, 4, 6], [7, 8, 9]]




```python
# Example 2 - Another example with 3x3 matrix
w, v = np.linalg.eig(arr4)
print("eigenvalues are {} and eigenvectors are {}".format(w, v))
```

    eigenvalues are [16.06225775 -2.         -0.06225775] and eigenvectors are [[-0.28409416 -0.66666667  0.58614342]
     [-0.48272522 -0.33333333 -0.77615549]
     [-0.82841226  0.66666667  0.2324189 ]]


In the above example, we calculated the eigenvalues and eigenvectors of a 3x3 matrix and stored these values in two variables w and v.  


```python
# Remember what was arr3
arr3
```




    [[1, 2, 4], [3, 4, 6]]




```python
# Example 3 - Let's try arr3, 2x3 array
np.linalg.eig(arr3)
```


    ---------------------------------------------------------------------------

    LinAlgError                               Traceback (most recent call last)

    /tmp/ipykernel_37/4246813020.py in <module>
          1 # Example 3 - Let's try arr3, 2x3 array
    ----> 2 np.linalg.eig(arr3)
    

    <__array_function__ internals> in eig(*args, **kwargs)


    /opt/conda/lib/python3.9/site-packages/numpy/linalg/linalg.py in eig(a)
       1314     a, wrap = _makearray(a)
       1315     _assert_stacked_2d(a)
    -> 1316     _assert_stacked_square(a)
       1317     _assert_finite(a)
       1318     t, result_t = _commonType(a)


    /opt/conda/lib/python3.9/site-packages/numpy/linalg/linalg.py in _assert_stacked_square(*arrays)
        201         m, n = a.shape[-2:]
        202         if m != n:
    --> 203             raise LinAlgError('Last 2 dimensions of the array must be square')
        204 
        205 def _assert_finite(*arrays):


    LinAlgError: Last 2 dimensions of the array must be square


Similar to det function, eig function is also working only for arrays which last two dimensions are square.

This is a powerful function to easily calculate eigenvectors and eigenvalues for an array.

## Function 3 - numpy.linalg.eigvals(a)

This is very similar to linalg.eig. The difference is eigvals gives only eigenvalues unlike eig, which also gives eigenvectors. 


```python
arr1
```




    [[1, 2], [3, 4.0]]




```python
# Example 1 - Application on arr1
np.linalg.eigvals(arr1)
```




    array([-0.37228132,  5.37228132])



Above you see, the function gave us the eigenvalues of the arr1.


```python
arr4
```




    [[1, 2, 4], [3, 4, 6], [7, 8, 9]]




```python
# Example 2 - Application on arr4
np.linalg.eigvals(arr4)
```




    array([16.06225775, -2.        , -0.06225775])



And it works for any square matrix. 3x3 case is above. 


```python
arr3
```




    [[1, 2, 4], [3, 4, 6]]




```python
# Example 3 - Application on arr3
np.linalg.eigvals(arr3)
```


    ---------------------------------------------------------------------------

    LinAlgError                               Traceback (most recent call last)

    /tmp/ipykernel_37/967568694.py in <module>
          1 # Example 3 - Application on arr3
    ----> 2 np.linalg.eigvals(arr3)
    

    <__array_function__ internals> in eigvals(*args, **kwargs)


    /opt/conda/lib/python3.9/site-packages/numpy/linalg/linalg.py in eigvals(a)
       1059     a, wrap = _makearray(a)
       1060     _assert_stacked_2d(a)
    -> 1061     _assert_stacked_square(a)
       1062     _assert_finite(a)
       1063     t, result_t = _commonType(a)


    /opt/conda/lib/python3.9/site-packages/numpy/linalg/linalg.py in _assert_stacked_square(*arrays)
        201         m, n = a.shape[-2:]
        202         if m != n:
    --> 203             raise LinAlgError('Last 2 dimensions of the array must be square')
        204 
        205 def _assert_finite(*arrays):


    LinAlgError: Last 2 dimensions of the array must be square


However, like previous functions, it does not work for non-square matrices for mathematical reasons. 

## Function 4 - numpy.linalg.inv(a)

This function calculates inverse of an array. 


```python
# Example 1 - Application on arr1
np.linalg.inv(arr1)
```




    array([[-2. ,  1. ],
           [ 1.5, -0.5]])



It gave us the inverse of the arr1. Let's check if it is really the inverse.


```python
inv_arr1=np.linalg.inv(arr1)
inv_arr1 @ arr1
```




    array([[1.00000000e+00, 0.00000000e+00],
           [1.11022302e-16, 1.00000000e+00]])



As you can see, the result is the unit matrix. So, it works.


```python
# Example 2 - Application on arr2
np.linalg.inv(arr2)
```




    array([[[-2. ,  1. ],
            [ 1.5, -0.5]],
    
           [[-4. ,  3. ],
            [ 3.5, -2.5]],
    
           [[-6. ,  5. ],
            [ 5.5, -4.5]]])



As you can see above, inverses of several matrices can be calculated together similar to other functions introduced. 


```python
arr5 = [[1, 2],
       [2, 4]]
```


```python
# Example 3 - Apllication on arr5
np.linalg.inv(arr5)
```


    ---------------------------------------------------------------------------

    LinAlgError                               Traceback (most recent call last)

    /tmp/ipykernel_37/1711048753.py in <module>
          1 # Example 3 - Apllication on arr5
    ----> 2 np.linalg.inv(arr5)
    

    <__array_function__ internals> in inv(*args, **kwargs)


    /opt/conda/lib/python3.9/site-packages/numpy/linalg/linalg.py in inv(a)
        543     signature = 'D->D' if isComplexType(t) else 'd->d'
        544     extobj = get_linalg_error_extobj(_raise_linalgerror_singular)
    --> 545     ainv = _umath_linalg.inv(a, signature=signature, extobj=extobj)
        546     return wrap(ainv.astype(result_t, copy=False))
        547 


    /opt/conda/lib/python3.9/site-packages/numpy/linalg/linalg.py in _raise_linalgerror_singular(err, flag)
         86 
         87 def _raise_linalgerror_singular(err, flag):
    ---> 88     raise LinAlgError("Singular matrix")
         89 
         90 def _raise_linalgerror_nonposdef(err, flag):


    LinAlgError: Singular matrix


As expected, it breaks because this matrix is singular, i.e., the first row is a linear combination of the second row.

This function is very convenient to calculate inverses of matrices.

## Function 5 - numpy.linalg.solve(a, b)

This function solves matrix equations. 

Let's say we have a system of equations as follows:  
5\*x0 + 2\*x1 = 10  
3\*x0 + 4\*x1 = 12  
To solve this we write the code below.


```python
# Example 1 - System of 2 eqns
arr7 = [[5, 2],
       [3, 4]]
arr8 = [10, 12]
np.linalg.solve(arr7, arr8)
```




    array([1.14285714, 2.14285714])



I gave us the array of [x0, x1] . Let's check if it holds.


```python
result1 =  np.linalg.solve(arr7, arr8)
arr7 @ result1
```




    array([10., 12.])



Yeah, it gave arr8 as expected. Let's now do an example with 3 eqns.  
Let's say we have a system of eqn's as follows:  
5\*x0 + 2\*x1 + x2 = 12  
x0 + x1 + 2\*x2 = 5  
7\*x0 - x1 + x2 = 5  


```python
# Example 2 - System of 3 eqns
arr12 = [[5, 2, 1],
        [1, 1, 2],
        [7, -1, 1]]
arr23 = [12, 5, 5]
result2 = np.linalg.solve(arr12, arr23)
result2
```




    array([1.09090909, 3.06060606, 0.42424242])



Again, it gave the result [x0, x1, x2] . Let's check if it is correct. 


```python
arr12 @ result2
```




    array([12.,  5.,  5.])



Yes, it works. Let's now do a last example with 2 eqns again.  
x0 + x1 = 12  
2\*x0 + 2\*x1 = 15 


```python
# Example 3 - Last example
a = [[1, 1],
    [2, 2]]
b = [12, 24]
result3 = np.linalg.solve(a, b)
result3
```


    ---------------------------------------------------------------------------

    LinAlgError                               Traceback (most recent call last)

    /tmp/ipykernel_37/3899491712.py in <module>
          3     [2, 2]]
          4 b = [12, 24]
    ----> 5 result3 = np.linalg.solve(a, b)
          6 result3


    <__array_function__ internals> in solve(*args, **kwargs)


    /opt/conda/lib/python3.9/site-packages/numpy/linalg/linalg.py in solve(a, b)
        391     signature = 'DD->D' if isComplexType(t) else 'dd->d'
        392     extobj = get_linalg_error_extobj(_raise_linalgerror_singular)
    --> 393     r = gufunc(a, b, signature=signature, extobj=extobj)
        394 
        395     return wrap(r.astype(result_t, copy=False))


    /opt/conda/lib/python3.9/site-packages/numpy/linalg/linalg.py in _raise_linalgerror_singular(err, flag)
         86 
         87 def _raise_linalgerror_singular(err, flag):
    ---> 88     raise LinAlgError("Singular matrix")
         89 
         90 def _raise_linalgerror_nonposdef(err, flag):


    LinAlgError: Singular matrix


As expected, it breaks. Look at the equations we put in. That equations have no unique solution. The array a is singular. We must use a non-singular array. 

This function can be used to solve matrix equations. 

## Conclusion

Numpy has very nice functions related to linear algebra.

## Reference Links
References and other interesting articles about Numpy arrays:
* Numpy official tutorial : https://numpy.org/doc/stable/user/quickstart.html
* https://numpy.org/doc/stable/reference/routines.linalg.html
