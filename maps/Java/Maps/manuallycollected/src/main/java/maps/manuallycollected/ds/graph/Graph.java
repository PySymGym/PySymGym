package maps.manuallycollected.ds.graph;

/**
* Form https://github.com/sherxon/AlgoDS/
*/

import java.util.Set;

public interface Graph {
     boolean addVertex(Integer t);

     Double addEdge(Integer from, Integer to);

     boolean addEdge(Integer from, Integer to, Double weight);

     boolean removeVertex(Integer t);

     boolean removeEdge(Integer from, Integer to);

     Set<Integer> getVertices();

     Set<Integer> getNeighbors(Integer ver);
     int size();
}
