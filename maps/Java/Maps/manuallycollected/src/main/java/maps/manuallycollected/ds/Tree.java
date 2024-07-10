package maps.manuallycollected.ds;

/**
* Form https://github.com/sherxon/AlgoDS/
*/

public interface Tree<K> {
    void insert(K k);
    boolean search(K k);
    void delete(K k);
}
