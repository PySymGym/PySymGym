package maps.manuallycollected;

/**
* Form https://github.com/sherxon/AlgoDS/
*/

import maps.manuallycollected.UnionFind;
import maps.manuallycollected.ds.graph.Edge;
import maps.manuallycollected.ds.graph.WeightedGraph;

import java.util.ArrayList;
import java.util.List;
import java.util.PriorityQueue;

/**
* This is another greedy algorithm to find Minimum Spanning Tree of a connected weighted Graph.
* The idea is choosing least weighted edge until all vertex are in the same connected component. In the beginning
* all vertices are considered as separate components and connect them by choosing least weighted edge.
* Union Find algorithm is used to keep track of connected components. Running time is O(E logE)
* */
public class KruskalsMST {

    private WeightedGraph graph;
    private UnionFind unionFind;
    public KruskalsMST(WeightedGraph graph) {
        this.graph = graph;
        this.unionFind=new UnionFind(graph.size());
    }

    public List<Edge> getMST() {
        List<Edge> list = new ArrayList<>();
        PriorityQueue<Edge> q = new PriorityQueue<>(graph.getEdges());
        while (!q.isEmpty()){
            Edge minEdge=q.remove(); // remove min Edge and check if both vertices of this edge is connected
            if(!unionFind.connected(minEdge.getFrom(), minEdge.getTo())){
                list.add(minEdge);
                unionFind.union(minEdge.getFrom(), minEdge.getTo()); // make both vertices one component
            }
        }
        return list;
    }
}
