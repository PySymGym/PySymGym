package maps.manuallycollected;

/**
* Form https://github.com/sherxon/AlgoDS/
*/

import maps.manuallycollected.ds.graph.Graph;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class TopologicalSorting extends DFS {
    List<Integer> list;

    public TopologicalSorting(Graph graph) {
        super(graph);
        list = new ArrayList<>();
    }

    // this works with DAG Only
    // first we will choose any vertex who that does not have incoming edges (sources)
    // sources can be found easier if incoming edge count is recorded in each vertex
    public List<Integer> topSort() {

        Set<Integer> sources = new HashSet<>();
        for (Integer vertex : graph.getVertices())
            sources.add(vertex);

        for (Integer vertex : graph.getVertices())
            for (Integer tVertex : graph.getNeighbors(vertex))
                sources.remove(tVertex);

        for (Integer source : sources)
            if (!visited.contains(source))
                search(source);

        return list;
    }

    @Override
    public void processVertex(Integer source) {
        super.processVertex(source);
        list.add(source);
    }
}
