package maps.manuallycollected;

/**
* Form https://github.com/sherxon/AlgoDS/
*/

public class InsertionSort {

    // sorts by shifting
    public static <T extends Comparable> void sort(T[] a) {
        for (int i = 1; i < a.length; i++) {
            for (int j = i; j > 0; j--) {
                if (a[j - 1].compareTo(a[j]) > 0) {
                    T temp = a[j - 1];
                    a[j - 1] = a[j];
                    a[j] = temp;
                }
            }
        }
    }

    // sorts by swapping
    public static <T extends Comparable> void sortBySwap(T[] a) {
        for (int i = 1; i < a.length; i++) {
            int j = i;
            T ii = a[i];

            while (j > 0 && a[j - 1].compareTo(ii) > 0) a[j] = a[--j];

            if (j != i) a[j] = ii;
        }
    }

    //simple version,  easy to understand.
    public static void sortSimple(int[] a) {

        for (int i = 1; i < a.length; i++) {
            for (int j = i; j > 0; j--) {
                if (a[j - 1] > a[j]) { //swap
                    int temp = a[j - 1];
                    a[j - 1] = a[j];
                    a[j] = temp;
                } else break;
            }
        }

    }


}
