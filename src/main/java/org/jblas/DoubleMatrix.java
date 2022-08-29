// --- BEGIN LICENSE BLOCK ---
/*
 * Copyright (c) 2009, Mikio L. Braun
 * Copyright (c) 2008, Johannes Schaback
 * Copyright (c) 2009, Jan Saputra Mueller
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *
 *     * Neither the name of the Technische Universitaet Berlin nor the
 *       names of its contributors may be used to endorse or promote
 *       products derived from this software without specific prior
 *       written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
// --- END LICENSE BLOCK ---
package org.jblas;

import lombok.val;
import org.jblas.exceptions.SizeException;
import org.jblas.ranges.Range;
import org.jblas.util.Random;

import java.io.*;

import java.util.AbstractList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.IntStream;

/**
 * A general matrix class for <tt>double</tt> typed values.
 * <p>
 * Don't be intimidated by the large number of methods this function defines. Most
 * are overloads provided for ease of use. For example, for each arithmetic operation,
 * up to six overloaded versions exist to handle in-place computations, and
 * scalar arguments (like adding a number to all elements of a matrix).
 *
 * <h3>Construction</h3>
 *
 * <p>To construct a two-dimensional matrices, you can use the following constructors
 * and static methods.</p>
 *
 * <table class="my">
 * <tr><th>Method<th>Description
 * <tr><td>DoubleMatrix(m,n, [value1, value2, value3...])<td>Values are filled in column by column.
 * <tr><td>DoubleMatrix(new double[][] {{value1, value2, ...}, ...}<td>Inner arrays are rows.
 * <tr><td>DoubleMatrix.zeros(m,n) <td>Initial values set to 0.0.
 * <tr><td>DoubleMatrix.ones(m,n) <td>Initial values set to 1.0.
 * <tr><td>DoubleMatrix.rand(m,n) <td>Values drawn at random between 0.0 and 1.0.
 * <tr><td>DoubleMatrix.randn(m,n) <td>Values drawn from normal distribution.
 * <tr><td>DoubleMatrix.eye(n) <td>Unit matrix (values 0.0 except for 1.0 on the diagonal).
 * <tr><td>DoubleMatrix.diag(array) <td>Diagonal matrix with given diagonal elements.
 * <caption>Matrix constructors.</caption>
 * </table>
 *
 * <p>Alternatively, you can construct (column) vectors, if you just supply the length
 * using the following constructors and static methods.</p>
 *
 * <table class="my">
 * <tr><th>Method</th>                        <th>Description</th></tr>
 * <tr><td>DoubleMatrix(m)</td>               <td>Constructs a column vector.</td></tr>
 * <tr><td>DoubleMatrix(new double[] {value1, value2, ...})</td><td>Constructs a column vector.</td></tr>
 * <tr><td>DoubleMatrix.zeros(m)</td>         <td>Initial values set to 0.0.</td></tr>
 * <tr><td>DoubleMatrix.ones(m)</td>          <td>Initial values set to 1.0.</td></tr>
 * <tr><td>DoubleMatrix.rand(m)</td>          <td>Values drawn at random between 0.0 and 1.0.</td></tr>
 * <tr><td>DoubleMatrix.randn(m)</td>         <td>Values drawn from normal distribution.</td></tr>
 * <tr><td>DoubleMatrix.linspace(a, b, n)</td><td>n linearly spaced values from a to b.</td></tr>
 * <tr><td>DoubleMatrix.logspace(a, b, n)</td><td>n logarithmically spaced values form 10^a to 10^b.</td></tr>
 * <caption>Column vector constructors.</caption>
 * </table>
 *
 * <p>You can also construct new matrices by concatenating matrices either horziontally
 * or vertically:</p>
 *
 * <table class="my">
 * <tr><th>Method<th>Description
 * <tr><td>x.concatHorizontally(y)<td>New matrix will be x next to y.
 * <tr><td>x.concatVertically(y)<td>New matrix will be x atop y.
 * <caption>Matrix concatenation.</caption>
 * </table>
 *
 * <h3>Element Access, Copying and Duplication</h3>
 *
 * <p>To access individual elements, or whole rows and columns, use the following
 * methods:<p>
 *
 * <table class="my">
 * <tr><th>x.Method<th>Description
 * <tr><td>x.get(i,j)<td>Get element in row i and column j.
 * <tr><td>x.put(i, j, v)<td>Set element in row i and column j to value v
 * <tr><td>x.get(i)<td>Get the ith element of the matrix (traversing rows first).
 * <tr><td>x.put(i, v)<td>Set the ith element of the matrix (traversing rows first).
 * <tr><td>x.getColumn(i)<td>Get a copy of column i.
 * <tr><td>x.putColumn(i, c)<td>Put matrix c into column i.
 * <tr><td>x.getRow(i)<td>Get a copy of row i.
 * <tr><td>x.putRow(i, c)<td>Put matrix c into row i.
 * <tr><td>x.swapColumns(i, j)<td>Swap the contents of columns i and j.
 * <tr><td>x.swapRows(i, j)<td>Swap the contents of rows i and j.
 * <caption>Element access.</caption>
 * </table>
 *
 * <p>For <tt>get</tt> and <tt>put</tt>, you can also pass integer arrays,
 * DoubleMatrix objects, or Range objects, which then specify the indices used
 * as follows:
 *
 * <ul>
 * <li><em>integer array:</em> the elements will be used as indices.
 * <li><em>DoubleMatrix object:</em> non-zero entries specify the indices.
 * <li><em>Range object:</em> see below.
 * </ul>
 *
 * <p>When using <tt>put</tt> with multiple indices, the assigned object must
 * have the correct size or be a scalar.</p>
 *
 * <p>There exist the following Range objects. The Class <tt>RangeUtils</tt> also
 * contains the a number of handy helper methods for constructing these ranges.</p>
 * <table class="my">
 * <tr><th>Class <th>RangeUtils method <th>Indices
 * <tr><td>AllRange <td>all() <td>All legal indices.
 * <tr><td>PointRange <td>point(i) <td> A single point.
 * <tr><td>IntervalRange <td>interval(a, b)<td> All indices from a to b (inclusive)
 * <tr><td rowspan=3>IndicesRange <td>indices(int[])<td> The specified indices.
 * <tr><td>indices(DoubleMatrix)<td>The specified indices.
 * <tr><td>find(DoubleMatrix)<td>The non-zero entries of the matrix.
 * <caption>Range objects.</caption>
 * </table>
 *
 * <p>The following methods can be used for duplicating and copying matrices.</p>
 *
 * <table class="my">
 * <tr><th>Method<th>Description
 * <tr><td>x.dup()<td>Get a copy of x.
 * <tr><td>x.copy(y)<td>Copy the contents of y to x (possible resizing x).
 * <caption>Copying matrices.</caption>
 * </table>
 *
 * <h3>Size and Shape</h3>
 *
 * <p>The following methods permit to access the size of a matrix and change its size or shape.</p>
 *
 * <table class="my">
 * <tr><th>x.Method<th>Description
 * <tr><td>x.rows<td>Number of rows.
 * <tr><td>x.columns<td>Number of columns.
 * <tr><td>x.length<td>Total number of elements.
 * <tr><td>x.isEmpty()<td>Checks whether rows == 0 and columns == 0.
 * <tr><td>x.isRowVector()<td>Checks whether rows == 1.
 * <tr><td>x.isColumnVector()<td>Checks whether columns == 1.
 * <tr><td>x.isVector()<td>Checks whether rows == 1 or columns == 1.
 * <tr><td>x.isSquare()<td>Checks whether rows == columns.
 * <tr><td>x.isScalar()<td>Checks whether length == 1.
 * <tr><td>x.resize(r, c)<td>Resize the matrix to r rows and c columns, discarding the content.
 * <tr><td>x.reshape(r, c)<td>Resize the matrix to r rows and c columns.<br> Number of elements must not change.
 * <caption>Size and size checks.</caption>
 * </table>
 *
 * <p>The size is stored in the <tt>rows</tt> and <tt>columns</tt> member variables.
 * The total number of elements is stored in <tt>length</tt>. Do not change these
 * values unless you know what you're doing!</p>
 *
 * <h3>Arithmetics</h3>
 *
 * <p>The usual arithmetic operations are implemented. Each operation exists in a
 * in-place version, recognizable by the suffix <tt>"i"</tt>, to which you can supply
 * the result matrix (or <tt>this</tt> is used, if missing). Using in-place operations
 * can also lead to a smaller memory footprint, as the number of temporary objects is
 * reduced (although the JVM garbage collector is usually pretty good at reusing these
 * temporary object immediately with little overhead.)</p>
 *
 * <p>Whenever you specify a result vector, the result vector must already have the
 * correct dimensions.</p>
 *
 * <p>For example, you can add two matrices using the <tt>add</tt> method. If you want
 * to store the result in of <tt>x + y</tt> in <tt>z</tt>, type
 * <span class=code>
 * x.addi(y, z)   // computes x = y + z.
 * </span>
 * Even in-place methods return the result, such that you can easily chain in-place methods,
 * for example:
 * <span class=code>
 * x.addi(y).addi(z) // computes x += y; x += z
 * </span></p>
 *
 * <p>Methods which operate element-wise only make sure that the length of the matrices
 * is correct. Therefore, you can add a 3 * 3 matrix to a 1 * 9 matrix, for example.</p>
 *
 * <p>Finally, there exist versions which take doubles instead of DoubleMatrix Objects
 * as arguments. These then compute the operation with the same value as the
 * right-hand-side. The same effect can be achieved by passing a DoubleMatrix with
 * exactly one element.</p>
 *
 * <table class="my">
 * <tr><th>Operation <th>Method <th>Comment
 * <tr><td>x + y <td>x.add(y)			<td>
 * <tr><td>x - y <td>x.sub(y), y.rsub(x) <td>rsub subtracts left from right hand side
 * <tr><td rowspan=3>x * y 	<td>x.mul(y) <td>element-wise multiplication
 * <tr>                     <td>x.mmul(y)<td>matrix-matrix multiplication
 * <tr>                     <td>x.dot(y) <td>scalar-product
 * <tr><td>x / y <td>x.div(y), y.rdiv(x) <td>rdiv divides right hand side by left hand side.
 * <tr><td>- x	 <td>x.neg()				<td>
 * <caption>Basic arithmetics.</caption>
 * </table>
 *
 * <p>There also exist operations which work on whole columns or rows.</p>
 *
 * <table class="my">
 * <tr><th>Method</th>           <th>Description</th></tr>
 * <tr><td>x.addRowVector</td>   <td>adds a vector to each row (addiRowVector works in-place)</td></tr>
 * <tr><td>x.addColumnVector</td><td>adds a vector to each column</td></tr>
 * <tr><td>x.subRowVector</td>   <td>subtracts a vector from each row</td></tr>
 * <tr><td>x.subColumnVector</td><td>subtracts a vector from each column</td></tr>
 * <tr><td>x.mulRowVector</td>   <td>Multiplies each row by a vector (elementwise)</td></tr>
 * <tr><td>x.mulColumnVector</td><td>Multiplies each column by a vector (elementwise)</td></tr>
 * <tr><td>x.divRowVector</td>   <td>Divide each row by a vector (elementwise)</td></tr>
 * <tr><td>x.divColumnVector</td><td>Divide each column by a vector (elementwise)</td></tr>
 * <tr><td>x.mulRow</td>         <td>Multiplies a row by a scalar</td></tr>
 * <tr><td>x.mulColumn</td>      <td>Multiplies a column by a scalar</td></tr>
 * <caption>Row and column arithmetics.</caption>
 * </table>
 *
 * <p>In principle, you could achieve the same result by first calling getColumn(),
 * adding, and then calling putColumn, but these methods are much faster.</p>
 *
 * <p>The following comparison operations are available</p>
 *
 * <table class="my">
 * <tr><th>Operation <th>Method
 * <tr><td>x &lt; y		<td>x.lt(y)
 * <tr><td>x &lt;= y	<td>x.le(y)
 * <tr><td>x &gt; y		<td>x.gt(y)
 * <tr><td>x &gt;= y	<td>x.ge(y)
 * <tr><td>x == y		<td>x.eq(y)
 * <tr><td>x != y		<td>x.ne(y)
 * <caption>Comparison operations.</caption>
 * </table>
 *
 * <p> Logical operations are also supported. For these operations, a value different from
 * zero is treated as "true" and zero is treated as "false". All operations are carried
 * out elementwise.</p>
 *
 * <table class="my">
 * <tr><th>Operation <th>Method
 * <tr><td>x &amp; y 	<td>x.and(y)
 * <tr><td>x | y 	<td>x.or(y)
 * <tr><td>x ^ y	<td>x.xor(y)
 * <tr><td>! x		<td>x.not()
 * <caption>Logical operations.</caption>
 * </table>
 *
 * <p>Finally, there are a few more methods to compute various things:</p>
 *
 * <table class="my">
 * <tr><th>Method <th>Description
 * <tr><td>x.max() <td>Return maximal element
 * <tr><td>x.argmax() <td>Return index of largest element
 * <tr><td>x.min() <td>Return minimal element
 * <tr><td>x.argmin() <td>Return index of smallest element
 * <tr><td>x.columnMins() <td>Return column-wise minima
 * <tr><td>x.columnArgmins() <td>Return column-wise index of minima
 * <tr><td>x.columnMaxs() <td>Return column-wise maxima
 * <tr><td>x.columnArgmaxs() <td>Return column-wise index of maxima
 * <caption>Minimum and maximum.</caption>
 * </table>
 *
 * @author Mikio Braun, Johannes Schaback
 */
@SuppressWarnings("unused")
public class DoubleMatrix extends ProtoMatrix implements Serializable, Cloneable {

    public static final DoubleMatrix EMPTY = new DoubleMatrix();

    // Precompile regex patterns
    private static final Pattern SEMICOLON = Pattern.compile(";");
    private static final Pattern WHITESPACES = Pattern.compile("\\s+");
    private static final Pattern COMMA = Pattern.compile(",");

    /**
     * Create a new matrix with <i>newRows</i> rows, <i>newColumns</i> columns
     * using <i>newData</i> as the data. Note that any change to the DoubleMatrix
     * will change the input array, too.
     *
     * @param newRows    the number of rows of the new matrix
     * @param newColumns the number of columns of the new matrix
     * @param newData    the data array to be used. Data must be stored by column (column-major)
     */
    public DoubleMatrix(int newRows, int newColumns, double... newData) {
        rows = newRows;
        columns = newColumns;
        length = rows * columns;

        if (newData != null && newData.length != newRows * newColumns)
            throw new IllegalArgumentException("Passed data must match matrix dimensions.");

        data = newData;
    }

    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>DoubleMatrix</tt>.
     *
     * @param newRows    the number of rows (<i>n</i>) of the new matrix.
     * @param newColumns the number of columns (<i>m</i>) of the new matrix.
     */
    public DoubleMatrix(int newRows, int newColumns) {
        this(newRows, newColumns, new double[newRows * newColumns]);
    }

    /**
     * Creates a new <tt>DoubleMatrix</tt> of size 0 times 0.
     */
    public DoubleMatrix() {
        this(0, 0, (double[]) null);
    }

    /**
     * Create a Matrix of length <tt>len</tt>. This creates a column vector.
     *
     * @param len vector length.
     */
    public DoubleMatrix(int len) {
        this(len, 1, new double[len]);
    }

    /**
     * Create a a column vector using <i>newData</i> as the data array.
     * Note that any change to the created DoubleMatrix will change in input array.
     */
    public DoubleMatrix(double[] newData) {
        this(newData.length, 1, newData);
    }

    /**
     * Creates a new matrix by reading it from a file.
     *
     * @param filename the path and name of the file to read the matrix from
     * @throws IOException when fails to read the file.
     */
    public DoubleMatrix(String filename) throws IOException {
        load(filename);
    }

    /**
     * Creates a new <i>n</i> times <i>m</i> <tt>DoubleMatrix</tt> from
     * the given <i>n</i> times <i>m</i> 2D data array. Note that the input array
     * is copied and any change to the DoubleMatrix will not change the input array.
     * The first dimension of the array makes the
     * rows (<i>n</i>) and the second dimension the columns (<i>m</i>). For example, the
     * given code <br><br>
     * <code>new DoubleMatrix(new double[][]{{1d, 2d, 3d}, {4d, 5d, 6d}, {7d, 8d, 9d}}).print();</code><br><br>
     * will constructs the following matrix:
     * <pre>
     * 1.0	2.0	3.0
     * 4.0	5.0	6.0
     * 7.0	8.0	9.0
     * </pre>.
     *
     * @param data <i>n</i> times <i>m</i> data array
     */
    public DoubleMatrix(double[][] data) {
        this(data.length, data[0].length);

        for (int r = 0; r < rows; r++) {
            assert (data[r].length == columns);
        }

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                put(r, c, data[r][c]);
            }
        }
    }

    /**
     * Creates a DoubleMatrix column vector from the given List&lt;Double&gt;.
     *
     * @param data data from which the entries are taken.
     */
    public DoubleMatrix(List<Double> data) {
        this(data.size());

        int c = 0;
        for (double d : data) {
            put(c++, d);
        }
    }

    /**
     * Construct DoubleMatrix from ASCII representation.
     * <p>
     * This is not very fast, but can be quiet useful when
     * you want to "just" construct a matrix, for example
     * when testing.
     * <p>
     * The format is semicolon separated rows of space separated values,
     * for example "1 2 3; 4 5 6; 7 8 9".
     */
    public static DoubleMatrix valueOf(String text) {
        String[] rowValues = SEMICOLON.split(text);

        DoubleMatrix result = null;

        // process rest
        for (int r = 0; r < rowValues.length; r++) {
            String[] columnValues = WHITESPACES.split(rowValues[r].trim());

            if (r == 0) {
                result = new DoubleMatrix(rowValues.length, columnValues.length);
            }

            for (int c = 0; c < columnValues.length; c++) {
                result.put(r, c, Double.parseDouble(columnValues[c]));
            }
        }

        return result;
    }

    /**
     * Create matrix with random values uniformly in 0..1.
     */
    public static DoubleMatrix rand(int rows, int columns) {
        DoubleMatrix m = new DoubleMatrix(rows, columns);

        for (int i = 0; i < rows * columns; i++) {
            m.data[i] = Random.nextDouble();
        }

        return m;
    }

    /**
     * Creates a column vector with random values uniformly in 0..1.
     */
    public static DoubleMatrix rand(int len) {
        return rand(len, 1);
    }

    /**
     * Create matrix with normally distributed random values.
     */
    public static DoubleMatrix randn(int rows, int columns) {
        DoubleMatrix m = new DoubleMatrix(rows, columns);

        for (int i = 0; i < rows * columns; i++) {
            m.data[i] = Random.nextGaussian();
        }

        return m;
    }

    /**
     * Create column vector with normally distributed random values.
     */
    public static DoubleMatrix randn(int len) {
        return randn(len, 1);
    }

    /**
     * Creates a new matrix in which all values are equal 0.
     */
    public static DoubleMatrix zeros(int rows, int columns) {
        return new DoubleMatrix(rows, columns);
    }

    /**
     * Creates a column vector of given length.
     */
    public static DoubleMatrix zeros(int length) {
        return zeros(length, 1);
    }

    /**
     * Creates a new matrix in which all values are equal 1.
     */
    public static DoubleMatrix ones(int rows, int columns) {
        DoubleMatrix m = new DoubleMatrix(rows, columns);

        for (int i = 0; i < rows * columns; i++) {
            m.put(i, 1.0);
        }

        return m;
    }

    /**
     * Creates a column vector with all elements equal to 1.
     */
    public static DoubleMatrix ones(int length) {
        return ones(length, 1);
    }

    /**
     * Construct a new n-by-n identity matrix.
     */
    public static DoubleMatrix eye(int n) {
        DoubleMatrix m = new DoubleMatrix(n, n);

        for (int i = 0; i < n; i++) {
            m.put(i, i, 1.0);
        }

        return m;
    }

    /**
     * Creates a new matrix where the values of the given vector are the diagonal values of
     * the matrix.
     */
    public static DoubleMatrix diag(DoubleMatrix x) {
        DoubleMatrix m = new DoubleMatrix(x.length, x.length);

        for (int i = 0; i < x.length; i++) {
            m.put(i, i, x.get(i));
        }

        return m;
    }

    /**
     * Construct a matrix of arbitrary shape and set the diagonal according
     * to a passed vector.
     * <p>
     * length of needs to be smaller than rows or columns.
     *
     * @param x       vector to fill the diagonal with
     * @param rows    number of rows of the resulting matrix
     * @param columns number of columns of the resulting matrix
     * @return a matrix with dimensions rows * columns whose diagonal elements are filled by x
     */
    public static DoubleMatrix diag(DoubleMatrix x, int rows, int columns) {
        DoubleMatrix m = new DoubleMatrix(rows, columns);

        for (int i = 0; i < x.length; i++) {
            m.put(i, i, x.get(i));
        }

        return m;
    }

    /**
     * Create a 1-by-1 matrix. For many operations, this matrix functions like a
     * normal double.
     */
    public static DoubleMatrix scalar(double s) {
        DoubleMatrix m = new DoubleMatrix(1, 1);
        m.put(0, 0, s);
        return m;
    }

    /**
     * Construct a column vector whose entries are logarithmically spaced points from
     * 10^lower to 10^upper using the specified number of steps
     *
     * @param lower starting exponent
     * @param upper ending exponent
     * @param size  number of steps
     * @return a column vector with (10^lower, ... 10^upper) with size many entries.
     */
    public static DoubleMatrix logspace(double lower, double upper, int size) {
        DoubleMatrix result = new DoubleMatrix(size);
        for (int i = 0; i < size; i++) {
            double t = (double) i / (size - 1);
            double e = lower * (1 - t) + t * upper;
            result.put(i, Math.pow(10.0, e));
        }
        return result;
    }

    /**
     * Construct a column vector whose entries are linearly spaced points from lower to upper with size
     * many steps.
     *
     * @param lower starting value
     * @param upper end value
     * @param size  number of steps
     * @return a column vector of size (lower, ..., upper) with size many entries.
     */
    public static DoubleMatrix linspace(int lower, int upper, int size) {
        DoubleMatrix result = new DoubleMatrix(size);
        for (int i = 0; i < size; i++) {
            double t = (double) i / (size - 1);
            result.put(i, lower * (1 - t) + t * upper);
        }
        return result;
    }

    /**
     * Concatenates two matrices horizontally. Matrices must have identical
     * numbers of rows.
     */
    public static DoubleMatrix concatHorizontally(DoubleMatrix A, DoubleMatrix B) {
        if (A.rows != B.rows) {
            throw new SizeException("Matrices don't have same number of rows.");
        }

        DoubleMatrix result = new DoubleMatrix(A.rows, A.columns + B.columns);
        SimpleBlas.copy(A, result);
        JavaBlas.rcopy(B.length, B.data, 0, 1, result.data, A.length, 1);
        return result;
    }

    /**
     * Concatenates two matrices vertically. Matrices must have identical
     * numbers of columns.
     */
    public static DoubleMatrix concatVertically(DoubleMatrix A, DoubleMatrix B) {
        if (A.columns != B.columns) {
            throw new SizeException("Matrices don't have same number of columns (" + A.columns + " != " + B.columns + ".");
        }

        DoubleMatrix result = new DoubleMatrix(A.rows + B.rows, A.columns);

        for (int i = 0; i < A.columns; i++) {
            JavaBlas.rcopy(A.rows, A.data, A.index(0, i), 1, result.data, result.index(0, i), 1);
            JavaBlas.rcopy(B.rows, B.data, B.index(0, i), 1, result.data, result.index(A.rows, i), 1);
        }

        return result;
    }

    public static DoubleMatrix loadAsciiFile(String filename) throws IOException {
        BufferedReader is = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));

        // Go through file and count columns and rows. What makes this endeavour a bit difficult is
        // that files can have leading or trailing spaces leading to spurious fields
        // after String.split().
        String line;
        int rows = 0;
        int columns = -1;
        while ((line = is.readLine()) != null) {
            String[] elements = WHITESPACES.split(line);
            int numElements = elements.length;
            if (elements[0].length() == 0) {
                numElements--;
            }
            if (elements[elements.length - 1].length() == 0) {
                numElements--;
            }

            if (columns == -1) {
                columns = numElements;
            } else {
                if (columns != numElements) {
                    throw new IOException("Number of elements changes in line " + line + ".");
                }
            }

            rows++;
        }
        is.close();

        try (FileInputStream fis = new FileInputStream(filename)) {
            // Go through file a second time process the actual data.
            is = new BufferedReader(new InputStreamReader(fis));
            DoubleMatrix result = new DoubleMatrix(rows, columns);
            int r = 0;
            while ((line = is.readLine()) != null) {
                String[] elements = WHITESPACES.split(line);
                int firstElement = (elements[0].length() == 0) ? 1 : 0;
                for (int c = 0, cc = firstElement; c < columns; c++, cc++) {
                    result.put(r, c, Double.parseDouble(elements[cc]));
                }
                r++;
            }
            return result;
        }
    }

    public static DoubleMatrix loadCSVFile(String filename) throws IOException {
        BufferedReader is = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));

        List<DoubleMatrix> rows = new LinkedList<DoubleMatrix>();
        String line;
        int columns = -1;
        while ((line = is.readLine()) != null) {
            String[] elements = COMMA.split(line);
            int numElements = elements.length;
            if (elements[0].length() == 0) {
                numElements--;
            }
            if (elements[elements.length - 1].length() == 0) {
                numElements--;
            }

            if (columns == -1) {
                columns = numElements;
            } else {
                if (columns != numElements) {
                    throw new IOException("Number of elements changes in line " + line + ".");
                }
            }

            DoubleMatrix row = new DoubleMatrix(columns);
            for (int c = 0; c < columns; c++) {
                row.put(c, Double.parseDouble(elements[c]));
            }
            rows.add(row);
        }
        is.close();

        DoubleMatrix result = new DoubleMatrix(rows.size(), columns);
        int r = 0;
        for (DoubleMatrix row : rows) {
            result.putRow(r, row);
            r++;
        }
        return result;
    }

    /**
     * Return the first element of the matrix.
     */
    public double scalar() {
        return get(0);
    }

    /**
     * Find the linear indices of all non-zero elements.
     */
    public int[] findIndices() {
        int len = 0;
        for (int i = 0; i < length; i++) {
            if (get(i) != 0.0) {
                len++;
            }
        }

        int[] indices = new int[len];
        int c = 0;

        for (int i = 0; i < length; i++) {
            if (get(i) != 0.0) {
                indices[c++] = i;
            }
        }

        return indices;
    }

    /**
     * Get all elements specified by the linear indices.
     */
    public DoubleMatrix get(int[] indices) {
        DoubleMatrix result = new DoubleMatrix(indices.length);

        for (int i = 0; i < indices.length; i++) {
            result.put(i, get(indices[i]));
        }

        return result;
    }

    /**
     * Get all elements for a given row and the specified columns.
     */
    public DoubleMatrix get(int r, int[] indices) {
        DoubleMatrix result = new DoubleMatrix(1, indices.length);

        for (int i = 0; i < indices.length; i++) {
            result.put(i, get(r, indices[i]));
        }

        return result;
    }

    /**
     * Get all elements for a given column and the specified rows.
     */
    public DoubleMatrix get(int[] indices, int c) {
        DoubleMatrix result = new DoubleMatrix(indices.length, 1);

        for (int i = 0; i < indices.length; i++) {
            result.put(i, get(indices[i], c));
        }

        return result;
    }

    /**
     * Get all elements from the specified rows and columns.
     */
    public DoubleMatrix get(int[] rindices, int[] cindices) {
        DoubleMatrix result = new DoubleMatrix(rindices.length, cindices.length);

        for (int i = 0; i < rindices.length; i++) {
            for (int j = 0; j < cindices.length; j++) {
                result.put(i, j, get(rindices[i], cindices[j]));
            }
        }

        return result;
    }

    /**
     * Get elements from specified rows and columns.
     */
    public DoubleMatrix get(Range rs, Range cs) {
        rs.init(0, rows);
        cs.init(0, columns);
        DoubleMatrix result = new DoubleMatrix(rs.length(), cs.length());

        for (; rs.hasMore(); rs.next()) {
            cs.init(0, columns);
            for (; cs.hasMore(); cs.next()) {
                result.put(rs.index(), cs.index(), get(rs.value(), cs.value()));
            }
        }

        return result;
    }

    public DoubleMatrix get(Range rs, int c) {
        rs.init(0, rows);
        DoubleMatrix result = new DoubleMatrix(rs.length(), 1);

        for (; rs.hasMore(); rs.next()) {
            result.put(rs.index(), 0, get(rs.value(), c));
        }

        return result;
    }

    public DoubleMatrix get(int r, Range cs) {
        cs.init(0, columns);
        DoubleMatrix result = new DoubleMatrix(1, cs.length());

        for (; cs.hasMore(); cs.next()) {
            result.put(0, cs.index(), get(r, cs.value()));
        }

        return result;

    }

    /**
     * Get elements specified by the non-zero entries of the passed matrix.
     */
    public DoubleMatrix get(DoubleMatrix indices) {
        return get(indices.findIndices());
    }

    /**
     * Get elements from a row and columns as specified by the non-zero entries of
     * a matrix.
     */
    public DoubleMatrix get(int r, DoubleMatrix indices) {
        return get(r, indices.findIndices());
    }

    /**
     * Get elements from a column and rows as specified by the non-zero entries of
     * a matrix.
     */
    public DoubleMatrix get(DoubleMatrix indices, int c) {
        return get(indices.findIndices(), c);
    }

    /**
     * Get elements from columns and rows as specified by the non-zero entries of
     * the passed matrices.
     */
    public DoubleMatrix get(DoubleMatrix rindices, DoubleMatrix cindices) {
        return get(rindices.findIndices(), cindices.findIndices());
    }

    /**
     * Return all elements with linear index a, a + 1, ..., b - 1.
     */
    public DoubleMatrix getRange(int a, int b) {
        DoubleMatrix result = new DoubleMatrix(b - a);

        for (int k = 0; k < b - a; k++) {
            result.put(k, get(a + k));
        }

        return result;
    }

    /**
     * Get elements from a row and columns <tt>a</tt> to <tt>b</tt>.
     */
    public DoubleMatrix getColumnRange(int r, int a, int b) {
        DoubleMatrix result = new DoubleMatrix(1, b - a);

        for (int k = 0; k < b - a; k++) {
            result.put(k, get(r, a + k));
        }

        return result;
    }

    /**
     * Get elements from a column and rows <tt>a</tt> to <tt>b</tt>.
     */
    public DoubleMatrix getRowRange(int a, int b, int c) {
        DoubleMatrix result = new DoubleMatrix(b - a);

        for (int k = 0; k < b - a; k++) {
            result.put(k, get(a + k, c));
        }

        return result;
    }

    /**
     * Get elements from rows <tt>ra</tt> to <tt>rb</tt> and
     * columns <tt>ca</tt> to <tt>cb</tt>.
     */
    public DoubleMatrix getRange(int ra, int rb, int ca, int cb) {
        DoubleMatrix result = new DoubleMatrix(rb - ra, cb - ca);

        for (int i = 0; i < rb - ra; i++) {
            for (int j = 0; j < cb - ca; j++) {
                result.put(i, j, get(ra + i, ca + j));
            }
        }

        return result;
    }

    /**
     * Get whole rows from the passed indices.
     */
    public DoubleMatrix getRows(int[] rindices) {
        DoubleMatrix result = new DoubleMatrix(rindices.length, columns);
        for (int i = 0; i < rindices.length; i++) {
            JavaBlas.rcopy(columns, data, index(rindices[i], 0), rows, result.data, result.index(i, 0), result.rows);
        }
        return result;
    }

    /**
     * Get whole rows as specified by the non-zero entries of a matrix.
     */
    public DoubleMatrix getRows(DoubleMatrix rindices) {
        return getRows(rindices.findIndices());
    }

    public DoubleMatrix getRows(Range indices, DoubleMatrix result) {
        indices.init(0, rows);
        if (result.rows < indices.length()) {
            throw new SizeException("Result matrix does not have enough rows (" + result.rows + " < " + indices.length() + ")");
        }
        result.checkColumns(columns);

        indices.init(0, rows);
        for (int r = 0; indices.hasMore(); indices.next(), r++) {
            for (int c = 0; c < columns; c++) {
                result.put(r, c, get(indices.value(), c));
            }
        }
        return result;
    }

    public DoubleMatrix getRows(Range indices) {
        indices.init(0, rows);
        DoubleMatrix result = new DoubleMatrix(indices.length(), columns);
        return getRows(indices, result);
    }

    /**
     * Get whole columns from the passed indices.
     */
    public DoubleMatrix getColumns(int[] cindices) {
        DoubleMatrix result = new DoubleMatrix(rows, cindices.length);
        for (int i = 0; i < cindices.length; i++) {
            JavaBlas.rcopy(rows, data, index(0, cindices[i]), 1, result.data, result.index(0, i), 1);
        }
        return result;
    }

    /**
     * Get whole columns as specified by the non-zero entries of a matrix.
     */
    public DoubleMatrix getColumns(DoubleMatrix cindices) {
        return getColumns(cindices.findIndices());
    }

    /**
     * Get whole columns as specified by Range.
     */
    public DoubleMatrix getColumns(Range indices, DoubleMatrix result) {
        indices.init(0, columns);
        if (result.columns < indices.length()) {
            throw new SizeException("Result matrix does not have enough columns (" + result.columns + " < " + indices.length() + ")");
        }
        result.checkRows(rows);

        indices.init(0, columns);
        for (int c = 0; indices.hasMore(); indices.next(), c++) {
            for (int r = 0; r < rows; r++) {
                result.put(r, c, get(r, indices.value()));
            }
        }
        return result;
    }

    public DoubleMatrix getColumns(Range indices) {
        indices.init(0, columns);
        DoubleMatrix result = new DoubleMatrix(rows, indices.length());
        return getColumns(indices, result);
    }

    /**
     * Set elements in linear ordering in the specified indices.
     * <p>
     * For example, <code>a.put(new int[]{ 1, 2, 0 }, new DoubleMatrix(3, 1, 2.0, 4.0, 8.0)</code>
     * does <code>a.put(1, 2.0), a.put(2, 4.0), a.put(0, 8.0)</code>.
     */
    public DoubleMatrix put(int[] indices, DoubleMatrix x) {
        if (x.isScalar()) {
            return put(indices, x.scalar());
        }
        x.checkLength(indices.length);

        for (int i = 0; i < indices.length; i++) {
            put(indices[i], x.get(i));
        }

        return this;
    }

    /**
     * Set multiple elements in a row.
     */
    public DoubleMatrix put(int r, int[] indices, DoubleMatrix x) {
        if (x.isScalar()) {
            return put(r, indices, x.scalar());
        }
        x.checkColumns(indices.length);

        for (int i = 0; i < indices.length; i++) {
            put(r, indices[i], x.get(i));
        }

        return this;
    }

    /**
     * Set multiple elements in a row.
     */
    public DoubleMatrix put(int[] indices, int c, DoubleMatrix x) {
        if (x.isScalar()) {
            return put(indices, c, x.scalar());
        }
        x.checkRows(indices.length);

        for (int i = 0; i < indices.length; i++) {
            put(indices[i], c, x.get(i));
        }

        return this;
    }

    /**
     * Put a sub-matrix as specified by the indices.
     */
    public DoubleMatrix put(int[] rindices, int[] cindices, DoubleMatrix x) {
        if (x.isScalar()) {
            return put(rindices, cindices, x.scalar());
        }
        x.checkRows(rindices.length);
        x.checkColumns(cindices.length);

        for (int i = 0; i < rindices.length; i++) {
            for (int j = 0; j < cindices.length; j++) {
                put(rindices[i], cindices[j], x.get(i, j));
            }
        }

        return this;
    }

    /**
     * Put a matrix into specified indices.
     */
    public DoubleMatrix put(Range rs, Range cs, DoubleMatrix x) {
        rs.init(0, rows);
        cs.init(0, columns);

        x.checkRows(rs.length());
        x.checkColumns(cs.length());

        for (; rs.hasMore(); rs.next()) {
            cs.init(0, columns);
            for (; cs.hasMore(); cs.next()) {
                put(rs.value(), cs.value(), x.get(rs.index(), cs.index()));
            }
        }

        return this;
    }

    /**
     * Put a single value into the specified indices (linear adressing).
     */
    public DoubleMatrix put(int[] indices, double v) {
        for (int index : indices) {
            put(index, v);
        }

        return this;
    }

    /**
     * Put a single value into a row and the specified columns.
     */
    public DoubleMatrix put(int r, int[] indices, double v) {
        for (int index : indices) {
            put(r, index, v);
        }

        return this;
    }

    /**
     * Put a single value into the specified rows of a column.
     */
    public DoubleMatrix put(int[] indices, int c, double v) {
        for (int index : indices) {
            put(index, c, v);
        }

        return this;
    }

    /**
     * Put a single value into the specified rows and columns.
     */
    public DoubleMatrix put(int[] rindices, int[] cindices, double v) {
        for (int rindex : rindices)
            for (int cindex : cindices) put(rindex, cindex, v);

        return this;
    }

    /**
     * Put a sub-matrix into the indices specified by the non-zero entries
     * of <tt>indices</tt> (linear adressing).
     */
    public DoubleMatrix put(DoubleMatrix indices, DoubleMatrix v) {
        return put(indices.findIndices(), v);
    }

    /**
     * Put a sub-vector into the specified columns (non-zero entries of <tt>indices</tt>) of a row.
     */
    public DoubleMatrix put(int r, DoubleMatrix indices, DoubleMatrix v) {
        return put(r, indices.findIndices(), v);
    }

    /**
     * Put a sub-vector into the specified rows (non-zero entries of <tt>indices</tt>) of a column.
     */
    public DoubleMatrix put(DoubleMatrix indices, int c, DoubleMatrix v) {
        return put(indices.findIndices(), c, v);
    }

    /**
     * Put a sub-matrix into the specified rows and columns (non-zero entries of
     * <tt>rindices</tt> and <tt>cindices</tt>.
     */
    public DoubleMatrix put(DoubleMatrix rindices, DoubleMatrix cindices, DoubleMatrix v) {
        return put(rindices.findIndices(), cindices.findIndices(), v);
    }

    /**
     * Put a single value into the elements specified by the non-zero
     * entries of <tt>indices</tt> (linear adressing).
     */
    public DoubleMatrix put(DoubleMatrix indices, double v) {
        return put(indices.findIndices(), v);
    }

    /**
     * Put a single value into the specified columns (non-zero entries of
     * <tt>indices</tt>) of a row.
     */
    public DoubleMatrix put(int r, DoubleMatrix indices, double v) {
        return put(r, indices.findIndices(), v);
    }

    /**
     * Put a single value into the specified rows (non-zero entries of
     * <tt>indices</tt>) of a column.
     */
    public DoubleMatrix put(DoubleMatrix indices, int c, double v) {
        return put(indices.findIndices(), c, v);
    }

    /**
     * Put a single value in the specified rows and columns (non-zero entries
     * of <tt>rindices</tt> and <tt>cindices</tt>.
     */
    public DoubleMatrix put(DoubleMatrix rindices, DoubleMatrix cindices, double v) {
        return put(rindices.findIndices(), cindices.findIndices(), v);
    }

    /**
     * Return transposed copy of this matrix.
     */
    public DoubleMatrix transpose() {
        DoubleMatrix result = new DoubleMatrix(columns, rows);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                result.put(j, i, get(i, j));
            }
        }

        return result;
    }

    /**
     * Compare two matrices. Returns true if and only if other is also a
     * DoubleMatrix which has the same size and the maximal absolute
     * difference in matrix elements is smaller than the specified tolerance
     */
    public boolean compare(Object o, double tolerance) {
        if (!(o instanceof DoubleMatrix other && sameSize(other))) return false;

        DoubleMatrix diff = MatrixFunctions.absi(sub(other));

        return diff.max() / (rows * columns) < tolerance;
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof DoubleMatrix other && sameSize(other) && Arrays.equals(data, other.data);
    }

    /**
     * Reshape the matrix in-place. Number of elements must not change.
     */
    public void reshape(int newRows, int newColumns) {
        if (length != newRows * newColumns) throw new IllegalArgumentException("Number of elements must not change.");

        rows = newRows;
        columns = newColumns;
    }

    /**
     * Generate a new matrix which has the given number of replications of this.
     */
    public DoubleMatrix repmat(int rowMult, int columnMult) {
        DoubleMatrix result = new DoubleMatrix(rows * rowMult, columns * columnMult);

        for (int c = 0; c < columnMult; c++) {
            for (int r = 0; r < rowMult; r++) {
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < columns; j++) {
                        result.put(r * rows + i, c * columns + j, get(i, j));
                    }
                }
            }
        }
        return result;
    }

    /**
     * Checks whether two matrices have the same size.
     */
    public boolean sameSize(DoubleMatrix a) {
        return rows == a.rows && columns == a.columns;
    }

    /**
     * Throws SizeException unless two matrices have the same size.
     */
    public void assertSameSize(DoubleMatrix a) {
        if (!sameSize(a)) {
            throw new SizeException("Matrices must have the same size.");
        }
    }

    /**
     * Checks whether two matrices can be multiplied (that is, number of columns of
     * this must equal number of rows of a.
     */
    public boolean multipliesWith(DoubleMatrix a) {
        return columns == a.rows;
    }

    /**
     * Throws SizeException unless matrices can be multiplied with one another.
     */
    public void assertMultipliesWith(DoubleMatrix a) {
        if (!multipliesWith(a)) {
            throw new SizeException("Number of columns of left matrix must be equal to number of rows of right matrix.");
        }
    }

    /**
     * Checks whether two matrices have the same length.
     */
    public boolean notSameLength(DoubleMatrix a) {
        return length != a.length;
    }

    /**
     * Throws SizeException unless matrices have the same length.
     */
    public void assertSameLength(DoubleMatrix a) {
        if (notSameLength(a)) {
            throw new SizeException("Matrices must have same length (is: " + length + " and " + a.length + ")");
        }
    }

    /**
     * Copy DoubleMatrix a to this. this a is resized if necessary.
     */
    public DoubleMatrix copy(DoubleMatrix a) {
        if (!sameSize(a)) {
            resize(a.rows, a.columns);
        }

        System.arraycopy(a.data, 0, data, 0, length);
        return a;
    }

    /**
     * Swap two columns of a matrix in-place.
     */
    public void swapColumns(int i, int j) {
        NativeBlas.dswap(rows, data, index(0, i), 1, data, index(0, j), 1);
    }

    /**
     * Swap two rows of a matrix in-place.
     */
    public void swapRows(int i, int j) {
        NativeBlas.dswap(columns, data, index(i, 0), rows, data, index(j, 0), rows);
    }

    /**
     * Set matrix element
     */
    public DoubleMatrix put(int rowIndex, int columnIndex, double value) {
        data[index(rowIndex, columnIndex)] = value;
        return this;
    }

    /**
     * Retrieve matrix element
     */
    public double get(int rowIndex, int columnIndex) {
        return data[index(rowIndex, columnIndex)];
    }

    /**
     * Get a matrix element (linear indexing).
     */
    public double get(int i) {
        return data[i];
    }

    /**
     * Set a matrix element (linear indexing).
     */
    public DoubleMatrix put(int i, double v) {
        data[i] = v;
        return this;
    }

    /**
     * Set all elements to a value.
     */
    public DoubleMatrix fill(double value) {
        for (int i = 0; i < length; i++) {
            put(i, value);
        }
        return this;
    }

    /**
     * Returns the diagonal of the matrix.
     */
    public DoubleMatrix diag() {
        assertSquare();
        DoubleMatrix d = new DoubleMatrix(rows);
        JavaBlas.rcopy(rows, data, 0, rows + 1, d.data, 0, 1);
        return d;
    }

    /**
     * Generate string representation of the matrix.
     */
    @Override
    public String toString() {
        return toString("%f");
    }

    /**
     * Generate string representation of the matrix, with specified
     * format for the entries. For example, <code>x.toString("%.1f")</code>
     * generates a string representations having only one position after the
     * decimal point.
     */
    public String toString(String fmt) {
        return toString(fmt, "[", "]", ", ", "; ");
    }

    /**
     * Generate string representation of the matrix, with specified
     * format for the entries, and delimiters.
     *
     * @param fmt    entry format (passed to String.format())
     * @param open   opening parenthesis
     * @param close  closing parenthesis
     * @param colSep separator between columns
     * @param rowSep separator between rows
     */
    public String toString(String fmt, String open, String close, String colSep, String rowSep) {
        StringWriter s = new StringWriter();
        PrintWriter p = new PrintWriter(s);

        p.print(open);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < columns; c++) {
                p.printf(fmt, get(r, c));
                if (c < columns - 1) {
                    p.print(colSep);
                }
            }
            if (r < rows - 1) {
                p.print(rowSep);
            }
        }

        p.print(close);

        return s.toString();
    }


    /**
     * Converts the matrix to a one-dimensional array of integers.
     */
    public int[] toIntArray() {
        return IntStream.range(0, length).map(i -> (int) Math.rint(get(i))).toArray();
    }

    /**
     * Convert the matrix to a two-dimensional array of integers.
     */
    public int[][] toIntArray2() {
        int[][] array = new int[rows][columns];

        for (int r = 0; r < rows; r++) for (int c = 0; c < columns; c++) array[r][c] = (int) Math.rint(get(r, c));

        return array;
    }

    /**
     * Convert the matrix to a one-dimensional array of boolean values.
     */
    public boolean[] toBooleanArray() {
        boolean[] array = new boolean[length];

        IntStream.range(0, length).forEach(i -> array[i] = get(i) != 0.0);

        return array;
    }

    /**
     * Convert the matrix to a two-dimensional array of boolean values.
     */
    public boolean[][] toBooleanArray2() {
        boolean[][] array = new boolean[rows][columns];

        for (int r = 0; r < rows; r++) for (int c = 0; c < columns; c++) array[r][c] = get(r, c) != 0.0;

        return array;
    }

    public List<Double> elementsAsList() {
        return new ElementsAsListView(this);
    }

    public List<DoubleMatrix> rowsAsList() {
        return new RowsAsListView(this);
    }

    public List<DoubleMatrix> columnsAsList() {
        return new ColumnsAsListView(this);
    }

    /**
     * Ensures that the result vector has the same length as this. If not,
     * resizing result is tried, which fails if result == this or result == other.
     */
    private void ensureResultLength(DoubleMatrix other, DoubleMatrix result) {
        if (notSameLength(result)) {
            if (result == this || result == other) {
                throw new SizeException("Cannot resize result matrix because it is used in-place.");
            }
            result.resize(rows, columns);
        }
    }


    /**
     * Add two matrices (in-place).
     */
    public void addi(DoubleMatrix other, DoubleMatrix result) {
        if (other.isScalar()) {
            addi(other.scalar(), result);
            return;
        }
        if (isScalar()) {
            other.addi(scalar(), result);
            return;
        }

        assertSameLength(other);
        ensureResultLength(other, result);

        if (result == this) {
            SimpleBlas.axpy(1.0, other, result);
        } else if (result == other) {
            SimpleBlas.axpy(1.0, this, result);
        } else {
            JavaBlas.rzgxpy(length, result.data, data, other.data);
        }
    }

    /**
     * Add a scalar to a matrix (in-place).
     */
    public void addi(double v, DoubleMatrix result) {
        ensureResultLength(null, result);

        for (int i = 0; i < length; i++) {
            result.put(i, get(i) + v);
        }
    }

    /**
     * Subtract two matrices (in-place).
     */
    public void subi(DoubleMatrix other, DoubleMatrix result) {
        if (other.isScalar()) {
            subi(other.scalar(), result);
            return;
        }
        if (isScalar()) {
            other.rsubi(scalar(), result);
            return;
        }

        assertSameLength(other);
        ensureResultLength(other, result);

        if (result == this) {
            SimpleBlas.axpy(-1.0, other, result);
        } else if (result == other) {
            SimpleBlas.scal(-1.0, result);
            SimpleBlas.axpy(1.0, this, result);
        } else {
            SimpleBlas.copy(this, result);
            SimpleBlas.axpy(-1.0, other, result);
        }
    }

    /**
     * Subtract a scalar from a matrix (in-place).
     */
    public void subi(double v, DoubleMatrix result) {
        ensureResultLength(null, result);

        for (int i = 0; i < length; i++) {
            result.put(i, get(i) - v);
        }
    }

    /**
     * Subtract two matrices, but subtract first from second matrix, that is,
     * compute <em>result = other - this</em> (in-place).
     */
    public void rsubi(DoubleMatrix other, DoubleMatrix result) {
        other.subi(this, result);
    }

    /**
     * Subtract a matrix from a scalar (in-place).
     */
    public void rsubi(double a, DoubleMatrix result) {
        ensureResultLength(null, result);

        for (int i = 0; i < length; i++) {
            result.put(i, a - get(i));
        }
    }

    /**
     * Elementwise multiplication (in-place).
     */
    public void muli(DoubleMatrix other, DoubleMatrix result) {
        if (other.isScalar()) {
            muli(other.scalar(), result);
            return;
        }
        if (isScalar()) {
            other.muli(scalar(), result);
            return;
        }

        assertSameLength(other);
        ensureResultLength(other, result);

        for (int i = 0; i < length; i++) {
            result.put(i, get(i) * other.get(i));
        }
    }

    /**
     * Elementwise multiplication with a scalar (in-place).
     */
    public void muli(double v, DoubleMatrix result) {
        ensureResultLength(null, result);

        for (int i = 0; i < length; i++) {
            result.put(i, get(i) * v);
        }
    }

    /**
     * Matrix-matrix multiplication (in-place).
     */
    public void mmuli(DoubleMatrix other, DoubleMatrix result) {
        if (other.isScalar()) {
            muli(other.scalar(), result);
            return;
        }
        if (isScalar()) {
            other.muli(scalar(), result);
            return;
        }

        /* check sizes and resize if necessary */
        assertMultipliesWith(other);
        if (result.rows != rows || result.columns != other.columns) {
            if (result != this && result != other) {
                result.resize(rows, other.columns);
            } else {
                throw new SizeException("Cannot resize result matrix because it is used in-place.");
            }
        }

        if (result == this || result == other) {
            /* actually, blas cannot do multiplications in-place. Therefore, we will fake by
             * allocating a temporary object on the side and copy the result later.
             */
            DoubleMatrix temp = new DoubleMatrix(result.rows, result.columns);
            if (other.columns == 1) {
                SimpleBlas.gemv(1.0, this, other, 0.0, temp);
            } else {
                SimpleBlas.gemm(1.0, this, other, 0.0, temp);
            }
            SimpleBlas.copy(temp, result);
        } else {
            if (other.columns == 1) {
                SimpleBlas.gemv(1.0, this, other, 0.0, result);
            } else {
                SimpleBlas.gemm(1.0, this, other, 0.0, result);
            }
        }
    }

    /**
     * Matrix-matrix multiplication with a scalar (for symmetry, does the
     * same as <code>muli(scalar)</code> (in-place).
     */
    public void mmuli(double v, DoubleMatrix result) {
        muli(v, result);
    }

    /**
     * Elementwise division (in-place).
     */
    public void divi(DoubleMatrix other, DoubleMatrix result) {
        if (other.isScalar()) {
            divi(other.scalar(), result);
            return;
        }
        if (isScalar()) {
            other.rdivi(scalar(), result);
            return;
        }

        assertSameLength(other);
        ensureResultLength(other, result);

        for (int i = 0; i < length; i++) {
            result.put(i, get(i) / other.get(i));
        }
    }

    /**
     * Elementwise division with a scalar (in-place).
     */
    public void divi(double a, DoubleMatrix result) {
        ensureResultLength(null, result);

        for (int i = 0; i < length; i++) {
            result.put(i, get(i) / a);
        }
    }

    /**
     * Elementwise division, with operands switched. Computes
     * <code>result = other / this</code> (in-place).
     */
    public void rdivi(DoubleMatrix other, DoubleMatrix result) {
        other.divi(this, result);
    }

    /**
     * (Elementwise) division with a scalar, with operands switched. Computes
     * <code>result = a / this</code> (in-place).
     */
    public void rdivi(double a, DoubleMatrix result) {
        ensureResultLength(null, result);

        for (int i = 0; i < length; i++) {
            result.put(i, a / get(i));
        }
    }

    /**
     * Negate each element (in-place).
     */
    public void negi() {
        for (int i = 0; i < length; i++) {
            put(i, -get(i));
        }
    }

    /**
     * Negate each element.
     */
    public DoubleMatrix neg() {
        DoubleMatrix scratch = clone();
        scratch.negi();
        return scratch;
    }

    /**
     * Maps zero to 1.0 and all non-zero values to 0.0 (in-place).
     */
    public void noti() {
        for (int i = 0; i < length; i++) {
            put(i, get(i) == 0.0 ? 1.0 : 0.0);
        }
    }

    /**
     * Maps zero to 1.0 and all non-zero values to 0.0.
     */
    public DoubleMatrix not() {
        DoubleMatrix scratch = clone();
        scratch.noti();
        return scratch;
    }

    /**
     * Maps zero to 0.0 and all non-zero values to 1.0 (in-place).
     */
    public void truthi() {
        for (int i = 0; i < length; i++) {
            put(i, get(i) == 0.0 ? 0.0 : 1.0);
        }
    }

    /**
     * Maps zero to 0.0 and all non-zero values to 1.0.
     */
    public DoubleMatrix truth() {
        DoubleMatrix scratch = clone();
        scratch.truthi();
        return scratch;
    }

    public void isNaNi() {
        for (int i = 0; i < length; i++) {
            put(i, Double.isNaN(get(i)) ? 1.0 : 0.0);
        }
    }

    public DoubleMatrix isNaN() {
        DoubleMatrix scratch = clone();
        scratch.isNaNi();
        return scratch;
    }

    public void isInfinitei() {
        for (int i = 0; i < length; i++) {
            put(i, Double.isInfinite(get(i)) ? 1.0 : 0.0);
        }
    }

    public DoubleMatrix isInfinite() {
        DoubleMatrix scratch = clone();
        scratch.isInfinitei();
        return scratch;
    }

    /**
     * Checks whether all entries (i, j) with i &gt;= j are zero.
     */
    public boolean isLowerTriangular() {
        for (int i = 0; i < rows; i++)
            for (int j = i + 1; j < columns; j++) {
                if (get(i, j) != 0.0)
                    return false;
            }

        return true;
    }

    /**
     * Checks whether all entries (i, j) with i &lt;= j are zero.
     */
    public boolean isUpperTriangular() {
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < i && j < columns; j++) {
                if (get(i, j) != 0.0)
                    return false;
            }

        return true;
    }

    public DoubleMatrix selecti(DoubleMatrix where) {// fixme finish rewriting in-place functions
        checkLength(where.length);
        for (int i = 0; i < length; i++) {
            if (where.get(i) == 0.0) {
                put(i, 0.0);
            }
        }
        return this;
    }

    public DoubleMatrix select(DoubleMatrix where) {
        return clone().selecti(where);
    }

    /**
     * Computes a rank-1-update A = A + alpha * x * y'.
     */
    public DoubleMatrix rankOneUpdate(double alpha, DoubleMatrix x, DoubleMatrix y) {
        if (rows != x.length) {
            throw new SizeException("Vector x has wrong length (" + x.length + " != " + rows + ").");
        }
        if (columns != y.length) {
            throw new SizeException("Vector y has wrong length (" + x.length + " != " + columns + ").");
        }

        SimpleBlas.ger(alpha, x, y, this);
        return this;
    }

    /**
     * Computes a rank-1-update A = A + alpha * x * x'.
     */
    public DoubleMatrix rankOneUpdate(double alpha, DoubleMatrix m) {
        return rankOneUpdate(alpha, m, m);
    }

    /**
     * Computes a rank-1-update A = A + x * x'.
     */
    public DoubleMatrix rankOneUpdate(DoubleMatrix m) {
        return rankOneUpdate(1.0, m, m);
    }

    /**
     * Computes a rank-1-update A = A + x * y'.
     */
    public DoubleMatrix rankOneUpdate(DoubleMatrix x, DoubleMatrix y) {
        return rankOneUpdate(1.0, x, y);
    }

    /**
     * Returns the minimal element of the matrix.
     */
    public double min() {
        if (isEmpty()) {
            return Double.POSITIVE_INFINITY;
        }
        double v = Double.POSITIVE_INFINITY;
        for (int i = 0; i < length; i++) {
            if (!Double.isNaN(get(i)) && get(i) < v) {
                v = get(i);
            }
        }

        return v;
    }

    /**
     * Returns the linear index of the minimal element. If there are more than one element with this value, the first
     * one is returned.
     */
    public int argmin() {
        if (isEmpty()) {
            return -1;
        }
        double v = Double.POSITIVE_INFINITY;
        int a = -1;
        for (int i = 0; i < length; i++) {
            if (!Double.isNaN(get(i)) && get(i) < v) {
                v = get(i);
                a = i;
            }
        }

        return a;
    }

    /**
     * Computes the minimum between two matrices. Returns the smaller of the
     * corresponding elements in the matrix (in-place).
     */
    public DoubleMatrix mini(DoubleMatrix other, DoubleMatrix result) {
        if (result == this) {
            for (int i = 0; i < length; i++) {
                if (get(i) > other.get(i)) {
                    put(i, other.get(i));
                }
            }
        } else {
            for (int i = 0; i < length; i++) {
                result.put(i, Math.min(get(i), other.get(i)));
            }
        }
        return result;
    }

    /**
     * Computes the minimum between two matrices. Returns the smaller of the
     * corresponding elements in the matrix (in-place on this).
     */
    public DoubleMatrix mini(DoubleMatrix other) {
        return mini(other, this);
    }

    /**
     * Computes the minimum between two matrices. Returns the smaller of the
     * corresponding elements in the matrix (in-place on this).
     */
    public DoubleMatrix min(DoubleMatrix other) {
        return mini(other, new DoubleMatrix(rows, columns));
    }

    public DoubleMatrix mini(double v, DoubleMatrix result) {
        if (result == this) {
            for (int i = 0; i < length; i++) {
                if (get(i) > v) {
                    result.put(i, v);
                }
            }
        } else {
            for (int i = 0; i < length; i++) {
                result.put(i, Math.min(get(i), v));
            }

        }
        return result;
    }

    public DoubleMatrix mini(double v) {
        return mini(v, this);
    }

    public DoubleMatrix min(double v) {
        return mini(v, new DoubleMatrix(rows, columns));
    }

    /**
     * Returns the maximal element of the matrix.
     */
    public double max() {
        if (isEmpty()) {
            return Double.NEGATIVE_INFINITY;
        }
        double v = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < length; i++) {
            if (!Double.isNaN(get(i)) && get(i) > v) {
                v = get(i);
            }
        }
        return v;
    }

    /**
     * Returns the linear index of the maximal element of the matrix. If
     * there are more than one elements with this value, the first one
     * is returned.
     */
    public int argmax() {
        if (isEmpty()) {
            return -1;
        }
        double v = Double.NEGATIVE_INFINITY;
        int a = -1;
        for (int i = 0; i < length; i++) {
            if (!Double.isNaN(get(i)) && get(i) > v) {
                v = get(i);
                a = i;
            }
        }

        return a;
    }

    /**
     * Computes the maximum between two matrices. Returns the larger of the
     * corresponding elements in the matrix (in-place).
     */
    public DoubleMatrix maxi(DoubleMatrix other, DoubleMatrix result) {
        if (result == this) {
            for (int i = 0; i < length; i++) {
                if (get(i) < other.get(i)) {
                    put(i, other.get(i));
                }
            }
        } else {
            for (int i = 0; i < length; i++) {
                result.put(i, Math.max(get(i), other.get(i)));
            }
        }
        return result;
    }

    /**
     * Computes the maximum between two matrices. Returns the smaller of the
     * corresponding elements in the matrix (in-place on this).
     */
    public DoubleMatrix maxi(DoubleMatrix other) {
        return maxi(other, this);
    } // fixme go through this part

    /**
     * Computes the maximum between two matrices. Returns the larger of the
     * corresponding elements in the matrix (in-place on this).
     */
    public DoubleMatrix max(DoubleMatrix other) {
        return maxi(other, new DoubleMatrix(rows, columns));
    }

    public DoubleMatrix maxi(double v, DoubleMatrix result) {
        if (result == this) {
            for (int i = 0; i < length; i++) {
                if (get(i) < v) {
                    result.put(i, v);
                }
            }
        } else {
            for (int i = 0; i < length; i++) {
                result.put(i, Math.max(get(i), v));
            }

        }
        return result;
    }

    public void maxi(double v) {
        maxi(v, this);
    }

    public DoubleMatrix max(double v) {
        return maxi(v, new DoubleMatrix(rows, columns));
    }

    /**
     * Computes the sum of all elements of the matrix.
     */
    public double sum() {
        double s = 0.0;
        for (int i = 0; i < length; i++) {
            s += get(i);
        }
        return s;
    }

    /**
     * Computes the product of all elements of the matrix
     */
    public double prod() {
        double p = 1.0;
        for (int i = 0; i < length; i++) {
            p *= get(i);
        }
        return p;
    }

    /**
     * Computes the mean value of all elements in the matrix,
     * that is, <code>x.sum() / x.length</code>.
     */
    public double mean() {
        return sum() / length;
    }

    /**
     * Computes the cumulative sum, that is, the sum of all elements
     * of the matrix up to a given index in linear addressing (in-place).
     */
    public DoubleMatrix cumulativeSumi() {
        double s = 0.0;
        for (int i = 0; i < length; i++) {
            s += get(i);
            put(i, s);
        }
        return this;
    }

    /**
     * Computes the cumulative sum, that is, the sum of all elements
     * of the matrix up to a given index in linear addressing.
     */
    public DoubleMatrix cumulativeSum() {
        return clone().cumulativeSumi();
    }

    /**
     * The scalar product of this with other.
     */
    public double dot(DoubleMatrix other) {
        return SimpleBlas.dot(this, other);
    }

    /**
     * Computes the projection coefficient of other on this.
     * <p>
     * The returned scalar times <tt>this</tt> is the orthogonal projection
     * of <tt>other</tt> on <tt>this</tt>.
     */
    public double project(DoubleMatrix other) {
        other.checkLength(length);
        double norm = 0, dot = 0;
        for (int i = 0; i < this.length; i++) {
            double x = get(i);
            norm += x * x;
            dot += x * other.get(i);
        }
        return dot / norm;
    }

    /**
     * The Euclidean norm of the matrix as vector, also the Frobenius
     * norm of the matrix.
     */
    public double norm2() {
        double norm = 0.0;
        for (int i = 0; i < length; i++) {
            norm += get(i) * get(i);
        }
        return Math.sqrt(norm);
    }

    /**
     * The maximum norm of the matrix (maximal absolute value of the elements).
     */
    public double normmax() {
        double max = 0.0;
        for (int i = 0; i < length; i++) {
            double a = Math.abs(get(i));
            if (a > max) {
                max = a;
            }
        }
        return max;
    }

    /**
     * The 1-norm of the matrix as vector (sum of absolute values of elements).
     */
    public double norm1() {
        double norm = 0.0;
        for (int i = 0; i < length; i++) {
            norm += Math.abs(get(i));
        }
        return norm;
    }

    /**
     * Returns the squared (Euclidean) distance.
     */
    public double squaredDistance(DoubleMatrix other) {
        other.checkLength(length);
        double sd = 0.0;
        for (int i = 0; i < length; i++) {
            double d = get(i) - other.get(i);
            sd += d * d;
        }
        return sd;
    }

    /**
     * Returns the (euclidean) distance.
     */
    public double distance2(DoubleMatrix other) {
        return Math.sqrt(squaredDistance(other));
    }

    /**
     * Returns the (1-norm) distance.
     */
    public double distance1(DoubleMatrix other) {
        other.checkLength(length);
        double d = 0.0;
        for (int i = 0; i < length; i++) {
            d += Math.abs(get(i) - other.get(i));
        }
        return d;
    }

    /**
     * Return a new matrix with all elements sorted.
     */
    public DoubleMatrix sort() {
        double[] array = toArray();
        java.util.Arrays.sort(array);
        return new DoubleMatrix(rows, columns, array);
    }

    /**
     * Sort elements in-place.
     */
    public DoubleMatrix sorti() {
        Arrays.sort(data);
        return this;
    }

    /**
     * Get the sorting permutation.
     *
     * @return an int[] array such that which indexes the elements in sorted
     * order.
     */
    public int[] sortingPermutation() {
        val indices = IntStream.range(0, length).boxed().toArray(Integer[]::new);

        val array = data;

        Arrays.sort(indices, (o1, o2) -> {
            int i = o1;
            int j = o2;
            return Double.compare(array[i], array[j]);
        });

        return Arrays.stream(indices).mapToInt(i -> i).toArray();
    }

    /**
     * Sort columns (in-place).
     */
    public void sortColumnsi() {
        for (int i = 0; i < length; i += rows) {
            Arrays.sort(data, i, i + rows);
        }
    }

    /**
     * Sort columns.
     */
    public DoubleMatrix sortColumns() {
        val scratch = this.clone();
        scratch.sortColumnsi();
        return scratch;
    }

    /**
     * Return matrix of indices which sort all columns.
     */
    public int[][] columnSortingPermutations() {
        int[][] result = new int[columns][];

        DoubleMatrix temp = new DoubleMatrix(rows);
        for (int c = 0; c < columns; c++) {
            result[c] = getColumn(c, temp).sortingPermutation();
        }

        return result;
    }

    /**
     * Sort rows (in-place).
     */
    public void sortRowsi() {
        // actually, this is much harder because the data is not consecutive
        // in memory...
        DoubleMatrix temp = new DoubleMatrix(columns);
        for (int r = 0; r < rows; r++) {
            putRow(r, getRow(r, temp).sorti());
        }
    }

    /**
     * Sort rows.
     */
    public DoubleMatrix sortRows() {
        val scratch = this.clone();
        scratch.sortRowsi();
        return scratch;
    }

    /**
     * Return matrix of indices which sort all columns.
     */
    public int[][] rowSortingPermutations() {
        int[][] result = new int[rows][];

        DoubleMatrix temp = new DoubleMatrix(columns);
        for (int r = 0; r < rows; r++) {
            result[r] = getRow(r, temp).sortingPermutation();
        }

        return result;
    }

    /**
     * Return a vector containing the sums of the columns (having number of columns many entries)
     */
    public DoubleMatrix columnSums() {
        if (rows == 1) {
            return clone();
        } else {
            DoubleMatrix v = new DoubleMatrix(1, columns);

            for (int c = 0; c < columns; c++) {
                for (int r = 0; r < rows; r++) {
                    v.put(c, v.get(c) + get(r, c));
                }
            }

            return v;
        }
    }

    /**
     * Return a vector containing the means of all columns.
     */
    public DoubleMatrix columnMeans() {
        var scratch = columnSums();
        columnSums().divi(rows);
        return scratch;
    }

    /**
     * Return a vector containing the sum of the rows.
     */
    public DoubleMatrix rowSums() {
        if (columns == 1) {
            return clone();
        } else {
            DoubleMatrix v = new DoubleMatrix(rows);

            for (int c = 0; c < columns; c++) {
                for (int r = 0; r < rows; r++) {
                    v.put(r, v.get(r) + get(r, c));
                }
            }

            return v;
        }
    }

    /**
     * Return a vector containing the means of the rows.
     */
    public DoubleMatrix rowMeans() {
        var scratch = rowSums();
        rowSums().divi(columns);
        return scratch;
    }

    /**
     * Get a copy of a column.
     */
    public DoubleMatrix getColumn(int c) {
        return getColumn(c, new DoubleMatrix(rows, 1));
    }

    /**
     * Copy a column to the given vector.
     */
    public DoubleMatrix getColumn(int c, DoubleMatrix result) {
        result.checkLength(rows);
        JavaBlas.rcopy(rows, data, index(0, c), 1, result.data, 0, 1);
        return result;
    }

    /**
     * Copy a column back into the matrix.
     */
    public void putColumn(int c, DoubleMatrix v) {
        JavaBlas.rcopy(rows, v.data, 0, 1, data, index(0, c), 1);
    }

    /**
     * Get a copy of a row.
     */
    public DoubleMatrix getRow(int r) {
        return getRow(r, new DoubleMatrix(1, columns));
    }

    /**
     * Copy a row to a given vector.
     */
    public DoubleMatrix getRow(int r, DoubleMatrix result) {
        result.checkLength(columns);
        JavaBlas.rcopy(columns, data, index(r, 0), rows, result.data, 0, 1);
        return result;
    }

    /**
     * Copy a row back into the matrix.
     */
    public void putRow(int r, DoubleMatrix v) {
        JavaBlas.rcopy(columns, v.data, 0, 1, data, index(r, 0), rows);
    }

    /**
     * Return column-wise minimums.
     */
    public DoubleMatrix columnMins() {
        DoubleMatrix mins = new DoubleMatrix(1, columns);
        for (int c = 0; c < columns; c++) {
            mins.put(c, getColumn(c).min());
        }
        return mins;
    }

    /**
     * Return index of minimal element per column.
     */
    public int[] columnArgmins() {
        int[] argmins = new int[columns];
        for (int c = 0; c < columns; c++) {
            argmins[c] = getColumn(c).argmin();
        }
        return argmins;
    }

    /**
     * Return column-wise maximums.
     */
    public DoubleMatrix columnMaxs() {
        DoubleMatrix maxs = new DoubleMatrix(1, columns);
        for (int c = 0; c < columns; c++) {
            maxs.put(c, getColumn(c).max());
        }
        return maxs;
    }

    /**
     * Return index of minimal element per column.
     */
    public int[] columnArgmaxs() {
        int[] argmaxs = new int[columns];
        for (int c = 0; c < columns; c++) {
            argmaxs[c] = getColumn(c).argmax();
        }
        return argmaxs;
    }

    /**
     * Return row-wise minimums.
     */
    public DoubleMatrix rowMins() {
        DoubleMatrix mins = new DoubleMatrix(rows);
        for (int c = 0; c < rows; c++) {
            mins.put(c, getRow(c).min());
        }
        return mins;
    }

    /**
     * Return index of minimal element per row.
     */
    public int[] rowArgmins() {
        int[] argmins = new int[rows];
        for (int c = 0; c < rows; c++) {
            argmins[c] = getRow(c).argmin();
        }
        return argmins;
    }

    /**
     * Return row-wise maximums.
     */
    public DoubleMatrix rowMaxs() {
        DoubleMatrix maxs = new DoubleMatrix(rows);
        for (int c = 0; c < rows; c++) {
            maxs.put(c, getRow(c).max());
        }
        return maxs;
    }

    /**
     * Return index of maximum element per row.
     */
    public int[] rowArgmaxs() {
        int[] argmaxs = new int[rows];
        for (int c = 0; c < rows; c++) {
            argmaxs[c] = getRow(c).argmax();
        }
        return argmaxs;
    }

    /**
     * Add a row vector to all rows of the matrix (in place).
     */
    public DoubleMatrix addiRowVector(DoubleMatrix x) {
        x.checkLength(columns);
        for (int c = 0; c < columns; c++) {
            for (int r = 0; r < rows; r++) {
                put(r, c, get(r, c) + x.get(c));
            }
        }
        return this;
    }

    /**
     * Add a row to all rows of the matrix.
     */
    public DoubleMatrix addRowVector(DoubleMatrix x) {
        return clone().addiRowVector(x);
    }

    /**
     * Add a vector to all columns of the matrix (in-place).
     */
    public DoubleMatrix addiColumnVector(DoubleMatrix x) {
        x.checkLength(rows);
        for (int c = 0; c < columns; c++) {
            for (int r = 0; r < rows; r++) {
                put(r, c, get(r, c) + x.get(r));
            }
        }
        return this;
    }

    /**
     * Add a vector to all columns of the matrix.
     */
    public DoubleMatrix addColumnVector(DoubleMatrix x) {
        return clone().addiColumnVector(x);
    }

    /**
     * Subtract a row vector from all rows of the matrix (in-place).
     */
    public DoubleMatrix subiRowVector(DoubleMatrix x) {
        // This is a bit crazy, but a row vector must have as length as the columns of the matrix.
        x.checkLength(columns);
        for (int c = 0; c < columns; c++) {
            for (int r = 0; r < rows; r++) {
                put(r, c, get(r, c) - x.get(c));
            }
        }
        return this;
    }

    /**
     * Subtract a row vector from all rows of the matrix.
     */
    public DoubleMatrix subRowVector(DoubleMatrix x) {
        return clone().subiRowVector(x);
    }

    /**
     * Subtract a column vector from all columns of the matrix (in-place).
     */
    public DoubleMatrix subiColumnVector(DoubleMatrix x) {
        x.checkLength(rows);
        for (int c = 0; c < columns; c++) {
            for (int r = 0; r < rows; r++) {
                put(r, c, get(r, c) - x.get(r));
            }
        }
        return this;
    }

    /**
     * Subtract a vector from all columns of the matrix.
     */
    public DoubleMatrix subColumnVector(DoubleMatrix x) {
        return clone().subiColumnVector(x);
    }

    /**
     * Multiply a row by a scalar.
     */
    public DoubleMatrix mulRow(int r, double scale) {
        NativeBlas.dscal(columns, scale, data, index(r, 0), rows);
        return this;
    }

    /**
     * Multiply a column by a scalar.
     */
    public DoubleMatrix mulColumn(int c, double scale) {
        NativeBlas.dscal(rows, scale, data, index(0, c), 1);
        return this;
    }

    /**
     * Multiply all columns with a column vector (in-place).
     */
    public DoubleMatrix muliColumnVector(DoubleMatrix x) {
        x.checkLength(rows);
        for (int c = 0; c < columns; c++) {
            for (int r = 0; r < rows; r++) {
                put(r, c, get(r, c) * x.get(r));
            }
        }
        return this;
    }

    /**
     * Multiply all columns with a column vector.
     */
    public DoubleMatrix mulColumnVector(DoubleMatrix x) {
        return clone().muliColumnVector(x);
    }

    /**
     * Multiply all rows with a row vector (in-place).
     */
    public DoubleMatrix muliRowVector(DoubleMatrix x) {
        x.checkLength(columns);
        for (int c = 0; c < columns; c++) {
            for (int r = 0; r < rows; r++) {
                put(r, c, get(r, c) * x.get(c));
            }
        }
        return this;
    }

    /**
     * Multiply all rows with a row vector.
     */
    public DoubleMatrix mulRowVector(DoubleMatrix x) {
        return clone().muliRowVector(x);
    }

    public DoubleMatrix diviRowVector(DoubleMatrix x) {
        x.checkLength(columns);
        for (int c = 0; c < columns; c++) {
            for (int r = 0; r < rows; r++) {
                put(r, c, get(r, c) / x.get(c));
            }
        }
        return this;
    }

    public DoubleMatrix divRowVector(DoubleMatrix x) {
        return clone().diviRowVector(x);
    }

    public DoubleMatrix diviColumnVector(DoubleMatrix x) {
        x.checkLength(rows);
        for (int c = 0; c < columns; c++) {
            for (int r = 0; r < rows; r++) {
                put(r, c, get(r, c) / x.get(r));
            }
        }
        return this;
    }

    public DoubleMatrix divColumnVector(DoubleMatrix x) {
        return clone().diviColumnVector(x);
    }

    /**
     * Saves this matrix to the specified file.
     *
     * @param filename the file to write the matrix in.
     * @throws IOException thrown on errors while writing the matrix to the file
     */
    public void save(String filename) throws IOException {
        FileOutputStream fos = new FileOutputStream(filename, false);
        try (fos; DataOutputStream dos = new DataOutputStream(fos)) {
            this.out(dos);
        }
    }

    /**
     * Loads a matrix from a file into this matrix. Note that the old data
     * of this matrix will be discarded.
     *
     * @param filename the file to read the matrix from
     * @throws IOException thrown on errors while reading the matrix
     */
    public void load(String filename) throws IOException {
        FileInputStream fis = new FileInputStream(filename);
        try (fis; DataInputStream dis = new DataInputStream(fis)) {
            this.in(dis);
        }
    }

    /**
     * Add a matrix (in place).
     */
    public void addi(DoubleMatrix other) {
        addi(other, this);
    }

    /**
     * Add a matrix.
     */
    public DoubleMatrix add(DoubleMatrix other) {
        var scratch = new DoubleMatrix(rows, columns);
        addi(other, scratch);
        return scratch;
    }

    /**
     * Add a scalar (in place).
     */
    public void addi(double v) {
        addi(v, this);
    }


    /**
     * Add a scalar.
     */
    public DoubleMatrix add(double v) {
        var scratch = new DoubleMatrix(rows, columns);
        addi(v, scratch);
        return scratch;
    }

    /**
     * Subtract a matrix (in place).
     */
    public void subi(DoubleMatrix other) {
        subi(other, this);
    }

    /**
     * Subtract a matrix.
     */
    public DoubleMatrix sub(DoubleMatrix other) {
        var scratch = new DoubleMatrix(rows, columns);
        subi(other, scratch);
        return scratch;
    }

    /**
     * Subtract a scalar (in place).
     */
    public void subi(double v) {
        subi(v, this);
    }

    /**
     * Subtract a scalar.
     */
    public DoubleMatrix sub(double v) {
        var scratch = new DoubleMatrix(rows, columns);
        subi(v, scratch);
        return scratch;
    }

    /**
     * (right-)subtract a matrix (in place).
     */
    public void rsubi(DoubleMatrix other) {
        rsubi(other, this);
    }

    /**
     * (right-)subtract a matrix.
     */
    public DoubleMatrix rsub(DoubleMatrix other) {
        var scratch = new DoubleMatrix(rows, columns);
        rsubi(other, scratch);
        return scratch;
    }

    /**
     * (right-)subtract a scalar (in place).
     */
    public void rsubi(double v) {
        rsubi(v, this);
    }

    /**
     * (right-)subtract a scalar.
     */
    public DoubleMatrix rsub(double v) {
        var scratch = new DoubleMatrix(rows, columns);
        rsubi(v, scratch);
        return scratch;
    }

    /**
     * Elementwise divide by a matrix (in place).
     */
    public void divi(DoubleMatrix other) {
        divi(other, this);
    }

    /**
     * Elementwise divide by a matrix.
     */
    public DoubleMatrix div(DoubleMatrix other) {
        var scratch = new DoubleMatrix(rows, columns);
        divi(other, scratch);
        return scratch;
    }

    /**
     * Elementwise divide by a scalar (in place).
     */
    public void divi(double v) {
        divi(v, this);
    }

    /**
     * Elementwise divide by a scalar.
     */
    public DoubleMatrix div(double v) {
        var scratch = new DoubleMatrix(rows, columns);
        divi(v, scratch);
        return scratch;
    }

    /**
     * (right-)elementwise divide by a matrix (in place).
     */
    public void rdivi(DoubleMatrix other) {
        rdivi(other, this);
    }

    /**
     * (right-)elementwise divide by a matrix.
     */
    public DoubleMatrix rdiv(DoubleMatrix other) {
        var scratch = new DoubleMatrix(rows, columns);
        rdivi(other, scratch);
        return scratch;
    }

    /**
     * (right-)elementwise divide by a scalar (in place).
     */
    public void rdivi(double v) {
        rdivi(v, this);
    }

    /**
     * (right-)elementwise divide by a scalar.
     */
    public DoubleMatrix rdiv(double v) {
        var scratch = new DoubleMatrix(rows, columns);
        rdivi(v, scratch);
        return scratch;
    }

    /**
     * Elementwise multiply by a matrix (in place).
     */
    public void muli(DoubleMatrix other) {
        muli(other, this);
    }

    /**
     * Elementwise multiply by a matrix.
     */
    public DoubleMatrix mul(DoubleMatrix other) {
        var scratch = new DoubleMatrix(rows, columns); // fixme if it's zero by default the result is zero too!
        muli(other, scratch);
        return scratch;
    }

    /**
     * Elementwise multiply by a scalar (in place).
     */
    public void muli(double v) {
        muli(v, this);
    }

    /**
     * Elementwise multiply by a scalar.
     */
    public DoubleMatrix mul(double v) {
        var scratch = new DoubleMatrix(rows, columns);
        muli(v, scratch);
        return scratch;
    }

    /**
     * Matrix-multiply by a matrix (in place).
     */
    public void mmuli(DoubleMatrix other) {
        mmuli(other, this);
    }

    /**
     * Matrix-multiply by a matrix.
     */
    public DoubleMatrix mmul(DoubleMatrix other) {
        // return mmuli(other, new DoubleMatrix(rows, other.columns)); //fixme validate parameters od the matrix
        var scratch = new DoubleMatrix(rows, columns);
        mmuli(other, scratch);
        return scratch;
    }

    /**
     * Matrix-multiply by a scalar (in place).
     */
    public void mmuli(double v) {
        mmuli(v, this);
    }

    /**
     * Matrix-multiply by a scalar.
     */
    public DoubleMatrix mmul(double v) {
        var scratch = new DoubleMatrix(rows, columns);
        mmuli(v, scratch);
        return scratch;
    }

    /**
     * Test for "less than" (in-place).
     */
    public void lti(DoubleMatrix other, DoubleMatrix result) {
        if (other.isScalar())
            lti(other.scalar(), result);

        assertSameLength(other);
        ensureResultLength(other, result);

        for (int i = 0; i < length; i++)
            result.put(i, get(i) < other.get(i) ? 1.0 : 0.0);
    }

    /**
     * Test for "less than" (in-place).
     */
    public void lti(DoubleMatrix other) {
        lti(other, this);
    }

    /**
     * Test for "less than".
     */
    public DoubleMatrix lt(DoubleMatrix other) {
        var scratch = new DoubleMatrix(rows, columns);
        lti(other, scratch);
        return scratch;
    }

    /**
     * Test for "less than" against a scalar (in-place).
     */
    public void lti(double value, DoubleMatrix result) {
        ensureResultLength(null, result);
        for (int i = 0; i < length; i++)
            result.put(i, get(i) < value ? 1.0 : 0.0);
    }

    /**
     * Test for "less than" against a scalar (in-place).
     */
    public void lti(double value) {
        lti(value, this);
    }

    /**
     * test for "less than" against a scalar.
     */
    public DoubleMatrix lt(double value) {
        var scratch = new DoubleMatrix(rows, columns);
        lti(value, scratch);
        return scratch;
    }

    /**
     * Test for "greater than" (in-place).
     */
    public void gti(DoubleMatrix other, DoubleMatrix result) {
        if (other.isScalar())
            gti(other.scalar(), result);

        assertSameLength(other);
        ensureResultLength(other, result);

        for (int i = 0; i < length; i++)
            result.put(i, get(i) > other.get(i) ? 1.0 : 0.0);
    }

    /**
     * Test for "greater than" (in-place).
     */
    public void gti(DoubleMatrix other) {
        gti(other, this);
    }

    /**
     * Test for "greater than".
     */
    public DoubleMatrix gt(DoubleMatrix other) {
        var scratch = new DoubleMatrix(rows, columns);
        gti(other, scratch);
        return scratch;
    }

    /**
     * Test for "greater than" against a scalar (in-place).
     */
    public void gti(double value, DoubleMatrix result) {
        ensureResultLength(null, result);
        for (int i = 0; i < length; i++)
            result.put(i, get(i) > value ? 1.0 : 0.0);
    }

    /**
     * Test for "greater than" against a scalar (in-place).
     */
    public void gti(double value) {
        gti(value, this);
    }

    /**
     * test for "greater than" against a scalar.
     */
    public DoubleMatrix gt(double value) {
        var scratch = new DoubleMatrix(rows, columns);
        gti(value, scratch);
        return scratch;
    }

    /**
     * Test for "less than or equal" (in-place).
     */
    public void lei(DoubleMatrix other, DoubleMatrix result) {
        if (other.isScalar())
            lei(other.scalar(), result);

        assertSameLength(other);
        ensureResultLength(other, result);

        for (int i = 0; i < length; i++)
            result.put(i, get(i) <= other.get(i) ? 1.0 : 0.0);
    }

    /**
     * Test for "less than or equal" (in-place).
     */
    public void lei(DoubleMatrix other) {
        lei(other, this);
    }

    /**
     * Test for "less than or equal".
     */
    public DoubleMatrix le(DoubleMatrix other) {
        var scratch = new DoubleMatrix(rows, columns);
        lei(other, scratch);
        return scratch;
    }

    /**
     * Test for "less than or equal" against a scalar (in-place).
     */
    public void lei(double value, DoubleMatrix result) {
        ensureResultLength(null, result);
        for (int i = 0; i < length; i++)
            result.put(i, get(i) <= value ? 1.0 : 0.0);
    }

    /**
     * Test for "less than or equal" against a scalar (in-place).
     */
    public void lei(double value) {
        lei(value, this);
    }

    /**
     * test for "less than or equal" against a scalar.
     */
    public DoubleMatrix le(double value) {
        var scratch = new DoubleMatrix(rows, columns);
        lei(value, scratch);
        return scratch;
    }

    /**
     * Test for "greater than or equal" (in-place).
     */
    public void gei(DoubleMatrix other, DoubleMatrix result) {
        if (other.isScalar())
            gei(other.scalar(), result);

        assertSameLength(other);
        ensureResultLength(other, result);

        for (int i = 0; i < length; i++)
            result.put(i, get(i) >= other.get(i) ? 1.0 : 0.0);
    }

    /**
     * Test for "greater than or equal" (in-place).
     */
    public void gei(DoubleMatrix other) {
        gei(other, this);
    }

    /**
     * Test for "greater than or equal".
     */
    public DoubleMatrix ge(DoubleMatrix other) {
        var scratch = new DoubleMatrix(rows, columns);
        gei(other, scratch);
        return scratch;
    }

    /**
     * Test for "greater than or equal" against a scalar (in-place).
     */
    public void gei(double value, DoubleMatrix result) {
        ensureResultLength(null, result);
        for (int i = 0; i < length; i++)
            result.put(i, get(i) >= value ? 1.0 : 0.0);
    }

    /**
     * Test for "greater than or equal" against a scalar (in-place).
     */
    public void gei(double value) {
        gei(value, this);
    }

    /**
     * test for "greater than or equal" against a scalar.
     */
    public DoubleMatrix ge(double value) {
        var scratch = new DoubleMatrix(rows, columns);
        gei(value, scratch);
        return scratch;
    }

    /**
     * Test for equality (in-place).
     */
    public void eqi(DoubleMatrix other, DoubleMatrix result) {
        if (other.isScalar())
            eqi(other.scalar(), result);

        assertSameLength(other);
        ensureResultLength(other, result);

        for (int i = 0; i < length; i++)
            result.put(i, get(i) == other.get(i) ? 1.0 : 0.0);
    }

    /**
     * Test for equality (in-place).
     */
    public void eqi(DoubleMatrix other) {
        eqi(other, this);
    }

    /**
     * Test for equality.
     */
    public DoubleMatrix eq(DoubleMatrix other) {
        var scratch = new DoubleMatrix(rows, columns);
        eqi(other, scratch);
        return scratch;
    }

    /**
     * Test for equality against a scalar (in-place).
     */
    public void eqi(double value, DoubleMatrix result) {
        ensureResultLength(null, result);
        for (int i = 0; i < length; i++)
            result.put(i, get(i) == value ? 1.0 : 0.0);
    }

    /**
     * Test for equality against a scalar (in-place).
     */
    public void eqi(double value) {
        eqi(value, this);
    }

    /**
     * test for equality against a scalar.
     */
    public DoubleMatrix eq(double value) {
        var scratch = new DoubleMatrix(rows, columns);
        eqi(value, scratch);
        return scratch;
    }

    /**
     * Test for inequality (in-place).
     */
    public void nei(DoubleMatrix other, DoubleMatrix result) {
        if (other.isScalar())
            nei(other.scalar(), result);

        assertSameLength(other);
        ensureResultLength(other, result);

        for (int i = 0; i < length; i++)
            result.put(i, get(i) != other.get(i) ? 1.0 : 0.0);
    }

    /**
     * Test for inequality (in-place).
     */
    public void nei(DoubleMatrix other) {
        nei(other, this);
    }

    /**
     * Test for inequality.
     */
    public DoubleMatrix ne(DoubleMatrix other) {
        var scratch = new DoubleMatrix(rows, columns);
        nei(other, scratch);
        return scratch;
    }

    /**
     * Test for inequality against a scalar (in-place).
     */
    public void nei(double value, DoubleMatrix result) {
        ensureResultLength(null, result);
        for (int i = 0; i < length; i++)
            result.put(i, get(i) != value ? 1.0 : 0.0);
    }

    /**
     * Test for inequality against a scalar (in-place).
     */
    public void nei(double value) {
        nei(value, this);
    }

    /**
     * test for inequality against a scalar.
     */
    public DoubleMatrix ne(double value) {
        var scratch = new DoubleMatrix(rows, columns);
        nei(value, scratch);
        return scratch;
    }

    /**
     * Compute elementwise logical and (in-place).
     */
    public void andi(DoubleMatrix other, DoubleMatrix result) {
        assertSameLength(other);
        ensureResultLength(other, result);

        for (int i = 0; i < length; i++)
            result.put(i, (get(i) != 0.0) & (other.get(i) != 0.0) ? 1.0 : 0.0);
    }

    /**
     * Compute elementwise logical and (in-place).
     */
    public void andi(DoubleMatrix other) {
        andi(other, this);
    }

    /**
     * Compute elementwise logical and.
     */
    public DoubleMatrix and(DoubleMatrix other) {
        var scratch = new DoubleMatrix(rows, columns);
        andi(other, scratch);
        return scratch;
    }

    /**
     * Compute elementwise logical and against a scalar (in-place).
     */
    public void andi(double value, DoubleMatrix result) {
        ensureResultLength(null, result);
        boolean val = (value != 0.0);
        for (int i = 0; i < length; i++)
            result.put(i, (get(i) != 0.0) & val ? 1.0 : 0.0);
    }

    /**
     * Compute elementwise logical and against a scalar (in-place).
     */
    public void andi(double value) {
        andi(value, this);
    }

    /**
     * Compute elementwise logical and against a scalar.
     */
    public DoubleMatrix and(double value) {
        var scratch = new DoubleMatrix(rows, columns);
        andi(value, scratch);
        return scratch;
    }

    /**
     * Compute elementwise logical or (in-place).
     */
    public void ori(DoubleMatrix other, DoubleMatrix result) {
        assertSameLength(other);
        ensureResultLength(other, result);

        for (int i = 0; i < length; i++)
            result.put(i, (get(i) != 0.0) | (other.get(i) != 0.0) ? 1.0 : 0.0);
    }

    /**
     * Compute elementwise logical or (in-place).
     */
    public void ori(DoubleMatrix other) {
        ori(other, this);
    }

    /**
     * Compute elementwise logical or.
     */
    public DoubleMatrix or(DoubleMatrix other) {
        var scratch = new DoubleMatrix(rows, columns);
        ori(other, scratch);
        return scratch;
    }

    /**
     * Compute elementwise logical or against a scalar (in-place).
     */
    public void ori(double value, DoubleMatrix result) {
        ensureResultLength(null, result);
        boolean val = (value != 0.0);
        for (int i = 0; i < length; i++)
            result.put(i, (get(i) != 0.0) | val ? 1.0 : 0.0);
    }

    /**
     * Compute elementwise logical or against a scalar (in-place).
     */
    public void ori(double value) {
        ori(value, this);
    }

    /**
     * Compute elementwise logical or against a scalar.
     */
    public DoubleMatrix or(double value) {
        var scratch = new DoubleMatrix(rows, columns);
        ori(value, scratch);
        return scratch;
    }

    /**
     * Compute elementwise logical xor (in-place).
     */
    public void xori(DoubleMatrix other, DoubleMatrix result) {
        assertSameLength(other);
        ensureResultLength(other, result);

        for (int i = 0; i < length; i++)
            result.put(i, (get(i) != 0.0) ^ (other.get(i) != 0.0) ? 1.0 : 0.0);
    }

    /**
     * Compute elementwise logical xor (in-place).
     */
    public void xori(DoubleMatrix other) {
        xori(other, this);
    }

    /**
     * Compute elementwise logical xor.
     */
    public DoubleMatrix xor(DoubleMatrix other) {
        var scratch = new DoubleMatrix(rows, columns);
        xori(other, scratch);
        return scratch;
    }

    /**
     * Compute elementwise logical xor against a scalar (in-place).
     */
    public void xori(double value, DoubleMatrix result) {
        ensureResultLength(null, result);
        boolean val = (value != 0.0);
        for (int i = 0; i < length; i++)
            result.put(i, (get(i) != 0.0) ^ val ? 1.0 : 0.0);
    }

    /**
     * Compute elementwise logical xor against a scalar (in-place).
     */
    public void xori(double value) {
        xori(value, this);
    }

    /**
     * Compute elementwise logical xor against a scalar.
     */
    public DoubleMatrix xor(double value) {
        var scratch = new DoubleMatrix(rows, columns);
        xori(value, scratch);
        return scratch;
    }

    public ComplexDoubleMatrix toComplex() {
        return new ComplexDoubleMatrix(this);
    }

    @Override
    public DoubleMatrix clone() {
        try {
            val clone = (DoubleMatrix) super.clone();
            clone.data = this.data.clone();
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new AssertionError();
        }
    }

    /**
     * A wrapper which allows to view a matrix as a List of Doubles (read-only!).
     * Also implements the {@link ConvertsToDoubleMatrix} interface.
     */
    public static class ElementsAsListView extends AbstractList<Double> implements ConvertsToDoubleMatrix {

        private final DoubleMatrix me;

        public ElementsAsListView(DoubleMatrix me) {
            this.me = me;
        }

        @Override
        public Double get(int index) {
            return me.get(index);
        }

        @Override
        public int size() {
            return me.length;
        }

        public DoubleMatrix convertToDoubleMatrix() {
            return me;
        }
    }

    public static class RowsAsListView extends AbstractList<DoubleMatrix> implements ConvertsToDoubleMatrix {

        private final DoubleMatrix me;

        public RowsAsListView(DoubleMatrix me) {
            this.me = me;
        }

        @Override
        public DoubleMatrix get(int index) {
            return me.getRow(index);
        }

        @Override
        public int size() {
            return me.rows;
        }

        public DoubleMatrix convertToDoubleMatrix() {
            return me;
        }
    }

    public static class ColumnsAsListView extends AbstractList<DoubleMatrix> implements ConvertsToDoubleMatrix {

        private final DoubleMatrix me;

        public ColumnsAsListView(DoubleMatrix me) {
            this.me = me;
        }

        @Override
        public DoubleMatrix get(int index) {
            return me.getColumn(index);
        }

        @Override
        public int size() {
            return me.columns;
        }

        public DoubleMatrix convertToDoubleMatrix() {
            return me;
        }
    }
}
