package maps.manuallycollected;

/**
* Form https://github.com/sherxon/AlgoDS/
*/

import maps.manuallycollected.ds.graph.Graph;

import java.util.Collections;
import java.util.List;
import java.util.Set;

public class BidirectionalBFS{


    List<Integer> searchBi(Graph graph, Integer source, Integer dest){
        BFSHelper sourceData=new BFSHelper(graph, source);
        BFSHelper destData=new BFSHelper(graph, dest);

        while (!sourceData.isDone() && !destData.isDone()) {

             Set<Integer> frontierSource=sourceData.searchLevel();
             Set<Integer> frontierDest=destData.searchLevel();

            for (Integer integer : frontierDest) {
                if(frontierSource.contains(integer))
                    return mergePath(sourceData, destData, integer);
            }
        }
        return null;
    }

    private List<Integer> mergePath(BFSHelper sourceData, BFSHelper destData, Integer integer) {
        List<Integer> pathFromSource = sourceData.getPath(integer);
        List<Integer> pathFromDest = destData.getPath(integer);
        Collections.reverse(pathFromSource);
        pathFromDest.remove(0);
        pathFromSource.addAll(pathFromDest);

        return pathFromSource;
    }

}
