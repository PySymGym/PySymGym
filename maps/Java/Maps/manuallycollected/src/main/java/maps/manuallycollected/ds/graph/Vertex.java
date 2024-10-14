package maps.manuallycollected.ds.graph;

/**
* Form https://github.com/sherxon/AlgoDS/
*/


import java.util.HashSet;
import java.util.Set;

public class Vertex<T> implements Comparable<Vertex<T>> {
    private T value;
    private Set<Vertex<T>> neighbors; // used with Unweighted graphs
    private Vertex<T> parent; // used in dfs and bfs
    private boolean visited; //used for bfs and dfs
    private Number weight;
    public Vertex(T value) {
        this.value = value;
        this.neighbors = new HashSet<>();
    }

    @Override
    public String toString() {
        return "Vertex{" +
                "value=" + value + '}';
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        Vertex<?> vertex = (Vertex<?>) o;

        return value.equals(vertex.value);

    }

    @Override
    public int hashCode() {
        return value.hashCode();
    }

    public Number getWeight() {
        return weight;
    }

    public void setWeight(Number weight) {
        this.weight = weight;
    }

    public void addNeighbor(Vertex<T> vertex){
        this.neighbors.add(vertex);
    }

    public T getValue() {
        return value;
    }

    public void setValue(T value) {
        this.value = value;
    }

    public Set<Vertex<T>> getNeighbors() {
        return neighbors;
    }


    public Vertex<T> getParent() {
        return parent;
    }

    public void setParent(Vertex parent) {
        this.parent = parent;
    }

    public boolean isVisited() {
        return visited;
    }

    public void setVisited(boolean visited) {
        this.visited = visited;
    }

    public void removeNeighrbor(Vertex<T> vertex) {
        this.neighbors.remove(vertex);
    }

    @Override
    public int compareTo(Vertex<T> o) {
        return (int) (this.weight.doubleValue() - o.weight.doubleValue());
    }
}
