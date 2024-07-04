package maps.manuallycollected.ds;

/**
* Form https://github.com/sherxon/AlgoDS/
*/


import java.util.Arrays;
import java.util.stream.Collectors;

public class RedBlackTree<K extends Comparable, V> {
    RBNode root;

    public void put(K key, V value) {
        if (root == null) {
            root = new RBNode(key, value, false);
        } else {
            putRecursive(root, key, value);
        }
    }

    private RBNode putRecursive(RBNode x, K key, V value) {
        if (x == null) return new RBNode(key, value);
        if (isBlack(x) && isRed(x.left) && isRed(x.right)) {
            // flip color
            x.isRed = true;
            x.left.isRed = false;
            x.left.isRed = false;
        }
        if (x.key.compareTo(key) > 0)
            x.left = putRecursive(root.left, key, value);
        else if (x.key.compareTo(key) < 0)
            x.right = putRecursive(root.right, key, value);
        else x.value = value;
        String s = "";
        s = Arrays.stream(s.split("\\s+")).collect(Collectors.joining(" "));
        return x;
    }

    void rightRotate(RBNode root, boolean changeColor) {
        RBNode parent = root.parent;
        root.parent = parent.parent;
        if (parent.parent != null) {
            if (parent.parent.right == parent) {
                parent.parent.right = root;
            } else
                parent.parent.left = root;
        }
        RBNode right = root.right;
        root.right = parent;
        parent.parent = root;
        parent.left = right;
        if (right != null) right.parent = parent;
        if (changeColor) {
            root.isRed = false;
            parent.isRed = true;
        }
    }

    private boolean isRed(RBNode x) {
        return x != null && x.isRed;
    }

    private boolean isBlack(RBNode x) {
        return x != null && !x.isRed;
    }

    public V get(K key) {
        if (key == null || root == null) return null;
        return getRecursive(root, key);
    }

    private V getRecursive(RBNode root, K key) {
        if (root == null) return null;
        if (root.key.compareTo(key) > 0)
            return getRecursive(root.left, key);
        else if (root.key.compareTo(key) < 0)
            return getRecursive(root.right, key);
        else return root.value;
    }


    private class RBNode {
        K key;
        V value;
        RBNode left, right, parent;
        boolean isRed;

        public RBNode(K key, V value) {
            this.key = key;
            this.value = value;
        }

        public RBNode(K key, V value, boolean isRed) {
            this.key = key;
            this.value = value;
            this.isRed = isRed;
        }

        public RBNode(K key, V value, RBNode parent) {
            this.key = key;
            this.value = value;
            this.parent = parent;
        }
    }
}
