#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 1024

// Struct to store node information
typedef struct {
    int id;
    double p[3];
} Node;

// Struct to store element information
typedef struct {
    int id;
    int node_ids[4];
} Element;

int main(void) {
    FILE *file = fopen("sphere.msh", "r");
    if (!file) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }

    char line[MAX_LINE_LENGTH];
    int num_nodes = 0, num_elements = 0;
    Node *nodes = NULL;
    Element *elements = NULL;

    size_t numEntityBlocks, numNodes, minNodeTag, maxNodeTag;

    while (fgets(line, MAX_LINE_LENGTH, file)) {
        // Check for Nodes section
        if (strstr(line, "$Nodes")) {
            // Read the number of nodes
            fgets(line, MAX_LINE_LENGTH, file);
            sscanf(line, "%lu %lu %lu %lu", &numEntityBlocks, &numNodes, &minNodeTag, &maxNodeTag);

            // Allocate memory for nodes
            nodes = (Node *)malloc(numNodes * sizeof(Node));
            int nodes_base_ptr = 0;

            // Read nodes
            for (int j = 0; j < numEntityBlocks; j++) {
                int entityDim, entityTag, parametric;
                size_t numNodesInBlock;
                fgets(line, MAX_LINE_LENGTH, file);
                sscanf(line, "%d %d %d %lu", &entityDim, &entityTag, &parametric, &numNodesInBlock);

                // Read node ids
                for (int i = 0; i < numNodesInBlock; i++) {
                    size_t id;
                    fgets(line, MAX_LINE_LENGTH, file);
                    sscanf(line, "%lu", &id);
                    nodes[nodes_base_ptr+i].id = id;
                }

                // Read node coords
                for (int i = 0; i < numNodesInBlock; i++) {
                    double x, y, z;
                    fgets(line, MAX_LINE_LENGTH, file);
                    sscanf(line, "%lf %lf %lf", &x, &y, &z);
                    nodes[nodes_base_ptr+i].p[0] = x;
                    nodes[nodes_base_ptr+i].p[1] = y;
                    nodes[nodes_base_ptr+i].p[2] = z;
                }

                nodes_base_ptr += numNodesInBlock;

                printf("Read (Dim:%d, Tag:%d): %lu nodes\n", entityDim, entityTag, numNodesInBlock);
            }

            num_nodes = numNodes;

        }

        // Check for Elements section
        if (strstr(line, "$Elements")) {
            size_t numEntityBlocks, numElements, minElementTag, maxElementTag;

            // Read the number of elements
            fgets(line, MAX_LINE_LENGTH, file);
            sscanf(line, "%ld %ld %ld %ld", &numEntityBlocks, &numElements, &minElementTag, &maxElementTag);

            // Read elements
            for (int j = 0; j < numEntityBlocks; j++) {
                int entityDim, entityTag, elementType;
                size_t numElementsInBlock;
                fgets(line, MAX_LINE_LENGTH, file);
                sscanf(line, "%d %d %d %lu", &entityDim, &entityTag, &elementType, &numElementsInBlock);
                printf("Read (Dim:%d, Tag:%d, Type: %d): %lu numElementsInBlock\n", entityDim, entityTag, elementType, numElementsInBlock);

                // we are only interested in tetrahedrons
                if (elementType != 4) {
                    for (int i = 0; i < numElementsInBlock; i++) {
                        fgets(line, MAX_LINE_LENGTH, file);
                    }
                    continue;
                }

                // Allocate memory for elements
                elements = (Element *)malloc(numElementsInBlock * sizeof(Element));

                // Read element ids & node ids
                for (int i = 0; i < numElementsInBlock; i++) {
                    size_t id, n1, n2, n3, n4;
                    fgets(line, MAX_LINE_LENGTH, file);
                    sscanf(line, "%lu %lu %lu %lu %lu", &id, &n1, &n2, &n3, &n4);
                    elements[i].id = id;
                    elements[i].node_ids[0] = n1;
                    elements[i].node_ids[1] = n2;
                    elements[i].node_ids[2] = n3;
                    elements[i].node_ids[3] = n4;
                }

                printf("Read: %lu elements\n", numElementsInBlock);

                num_elements = numElementsInBlock;

            }

        }
    }

    fclose(file);

    // Output loaded nodes and elements (for verification)
    printf("Loaded %d nodes:\n", num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        printf("Node %d: (%lf, %lf, %lf)\n", nodes[i].id, nodes[i].p[0], nodes[i].p[1], nodes[i].p[2]);
    }

    printf("\nLoaded %d tetrahedral elements:\n", num_elements);
    for (int i = 0; i < num_elements; i++) {
        printf("Element %d: Nodes [%d, %d, %d, %d]\n", elements[i].id, elements[i].node_ids[0],
               elements[i].node_ids[1], elements[i].node_ids[2], elements[i].node_ids[3]);
    }

    // for (int i = 0; i < num_elements; i++) {
    //     real_t *p0 = nodes[elements[i].node_ids[0]];
    //     real_t *p1 = nodes[elements[i].node_ids[1]];
    //     real_t *p2 = nodes[elements[i].node_ids[2]];
    //     real_t *p3 = nodes[elements[i].node_ids[3]];
    //     compute_A(p0, p1, p2, p3, macro_J);

    //     solve_using_gradient_descent(tetra_level, nodes, tets, macro_J);
    // }


    // Free allocated memory
    free(nodes);
    free(elements);

    return 0;
}
