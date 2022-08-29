package org.jblas;

import org.jblas.exceptions.SizeException;

import java.io.*;
import java.util.Arrays;

abstract class ProtoMatrix implements Serializable {
    /**
     * Total number of elements (for convenience).
     */
    public int length;
    /**
     * Number of rows.
     */
    public int rows;
    /**
     * Number of columns.
     */
    public int columns;

    /**
     * The actual data stored by rows (that is, row 0, row 1...).
     */
    public double[] data = null; // rows are contiguous// fixme doesn't work with flaats

    /**
     * Serialization
     */
    @Serial
    private void writeObject(ObjectOutputStream s) throws IOException {
        s.defaultWriteObject();
    }

    @Serial
    private void readObject(ObjectInputStream s) throws IOException, ClassNotFoundException {
        s.defaultReadObject();
    }


    /**
     * Test whether a matrix is scalar.
     */
    public boolean isScalar() {
        return length == 1;
    }

    /**
     * Assert that the matrix has a certain length.
     *
     * @throws SizeException
     */
    public void checkLength(int l) {
        if (length != l) {
            throw new SizeException("Matrix does not have the necessary length (" + length + " != " + l + ").");
        }
    }

    /**
     * Get index of an element
     */
    public int index(int rowIndex, int columnIndex) {
        return rowIndex + rows * columnIndex;
    }

    /**
     * Compute the row index of a linear index.
     */
    public int indexRows(int i) {
        return i - indexColumns(i) * rows;
    }

    /**
     * Compute the column index of a linear index.
     */
    public int indexColumns(int i) {
        return i / rows;
    }

    /**
     * Get number of rows.
     */
    public int getRows() {
        return rows;
    }

    /**
     * Get number of columns.
     */
    public int getColumns() {
        return columns;
    }

    /**
     * Get total number of elements.
     */
    public int getLength() {
        return length;
    }

    /**
     * Checks whether the matrix is empty.
     */
    public boolean isEmpty() {
        return columns == 0 || rows == 0;
    }

    /**
     * Checks whether the matrix is square.
     */
    public boolean isSquare() {
        return columns == rows;
    }

    /**
     * Throw SizeException unless matrix is square.
     */
    public void assertSquare() {
        if (!isSquare()) {
            throw new SizeException("Matrix must be square!");
        }
    }

    /**
     * Checks whether the matrix is a vector.
     */
    public boolean isVector() {
        return columns == 1 || rows == 1;
    }

    /**
     * Checks whether the matrix is a row vector.
     */
    public boolean isRowVector() {
        return rows == 1;
    }

    /**
     * Checks whether the matrix is a column vector.
     */
    public boolean isColumnVector() {
        return columns == 1;
    }


    /**
     * Asserts that the matrix has a certain number of rows.
     *
     * @throws SizeException
     */
    public void checkRows(int r) {
        if (rows != r) {
            throw new SizeException("Matrix does not have the necessary number of rows (" + rows + " != " + r + ").");
        }
    }

    /**
     * Asserts that the amtrix has a certain number of columns.
     *
     * @throws SizeException
     */
    public void checkColumns(int c) {
        if (columns != c) {
            throw new SizeException("Matrix does not have the necessary number of columns (" + columns + " != " + c + ").");
        }
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 83 * hash + this.rows;
        hash = 83 * hash + this.columns;
        hash = 83 * hash + Arrays.hashCode(this.data);
        return hash;
    }


    /**
     * Resize the matrix. All elements will be set to zero.
     */
    public void resize(int newRows, int newColumns) {
        rows = newRows;
        columns = newColumns;
        length = newRows * newColumns;
        data = new double[rows * columns];
    }

    /**
     * Pretty-print this matrix to <tt>System.out</tt>.
     */
    public void print() {
        System.out.println(this);
    }

    /**
     * Converts the matrix to a one-dimensional array of doubles.
     */
    public double[] toArray() {
        double[] array = new double[length];

        System.arraycopy(data, 0, array, 0, length);

        return array;
    }


    /**
     * Converts the matrix to a two-dimensional array of doubles.
     */
    public double[][] toArray2() {
        double[][] array = new double[rows][columns];
        for (int r = 0; r < rows; r++) System.arraycopy(data, r * columns, array[r], 0, columns);
        return array;
    }

    /**
     * Reads in a matrix from the given data stream. Note
     * that the old data of this matrix will be discarded.
     *
     * @param dis the data input stream to read from.
     * @throws IOException
     */
    public void in(DataInputStream dis) throws IOException {
        if (!dis.readUTF().equals("double")) {
            throw new IllegalStateException("The matrix in the specified file is not of the correct type!");
        }

        this.columns = dis.readInt();
        this.rows = dis.readInt();

        final int MAX = dis.readInt();
        data = new double[MAX];
        for (int i = 0; i < MAX; i++) {
            data[i] = dis.readDouble();
        }
    }


    /**
     * Writes out this matrix to the given data stream.
     *
     * @param dos the data output stream to write to.
     * @throws IOException
     */
    public void out(DataOutputStream dos) throws IOException {
        dos.writeUTF("double");
        dos.writeInt(columns);
        dos.writeInt(rows);

        dos.writeInt(data.length);
        for (double datum : data) {
            dos.writeDouble(datum);
        }
    }

    public <T> void fromArrayToList(T[] a) {

    }

}
