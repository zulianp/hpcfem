import numpy as np
import math, os

macro_tets = 1

shown = [False for i in range(6)]
lapl_shown = [False for i in range(6)]

def tetrahedron_volume(A, B, C, D):
    # Create matrix with columns as vectors AB, AC, and AD
    mat = np.array([B - A, C - A, D - A])
    volume = np.linalg.det(mat) / 6.0
    return volume

def compute_A(p0, p1, p2, p3, category=0) :
    u = p1 - p0
    v = p2 - p0
    w = p3 - p0

    print(category)

    A = np.zeros((3, 3))
    A[:,0] = u
    A[:,1] = v
    A[:,2] = w

    if not shown[category] :
        print(A)
        shown[category] = True

    ## print("A =", A, np.linalg.det(A))
    assert(np.linalg.det(A) > 0)
    return A

def create_tetn_mesh(nodes):
    in_list = list()
    for i in range(nodes) :
        in_list.append(np.array([i]))
    return in_list

def compute_nodes_number(tetra_level) :
    nodes = 0
    if tetra_level % 2 == 0 :
        for i in range(0, tetra_level//2) :
            nodes += (tetra_level-i+1) * (i+1) * 2
            print(tetra_level-i+1, i+1)
        nodes += (tetra_level//2+1) * (tetra_level//2+1)
    else :
        for i in range(0, tetra_level//2+1) :
            nodes += (tetra_level-i+1) * (i+1) * 2
            print(tetra_level-i+1, i+1)
    return nodes

def create_macro_tet4_mesh(tetra_level):
    nodes = compute_nodes_number(tetra_level)
    return create_tetn_mesh(nodes)

def check_laplacian_matrix(L):
    """
    Check the Laplacian matrix for the following properties:
    1. Each row sum is zero.
    2. The matrix is symmetric.
    3. The nonzero diagonal entries have opposite signs to their corresponding off-diagonal entries.
    """
    
    # Check if the matrix is square
    assert L.shape[0] == L.shape[1], "Matrix must be square"
    
    n = L.shape[0]

    # Check that the row sum is zero
    row_sums = np.sum(L, axis=1)
    assert np.allclose(row_sums, 0), "Row sums are not zero"

    # Check that the matrix is symmetric
    assert np.allclose(L, L.T), "Matrix is not symmetric"

    # print(L)

    # All off diag entries have the opposite sign of the diag entries
    # Check that the nonzero diagonal entries have opposite signs to their corresponding off-diagonal entries
    # for i in range(n):
    #     diag_entry = L[i, i]
    #     off_diag_entries = L[i, np.arange(n) != i]
        
    #     if diag_entry != 0:
    #         assert np.all((off_diag_entries >= 0) if diag_entry < 0 else (off_diag_entries <= 0)), \
    #             f"Diagonal entry {diag_entry} at index {i} does not have opposite sign to its off-diagonal entries"

    # print("Laplacian matrix is valid")

# put a bp at x, after 1st time iter, neighbors of 1 should have non zero values after mult; should be negative  

# takes micro_elems (i0 to ik), compute_M/compute_Lapl as arg
def assemble_macro_elem(micro_elems, tetra_level, nodes, x_coords, y_coords, z_coords, computeFn, vecX) :
    vecY = np.zeros((len(x_coords), 1))
    level = tetra_level + 1

    n_macro_elems = 1
    dofs = list()

    i0 = list()
    i1 = list()
    i2 = list()
    i3 = list()
    category = list()

    for elem in range(0, n_macro_elems) :
        
        for i in range(nodes) :
            # dofs contain the index of nodes of each macro elements
            # [[0, ...], [1], [2], ...]
            dofs.append(micro_elems[i][elem])
                
        # Build sub-elements
        micro_tets = list()
        p = 0

        for i in range(0, level-1) :
            layer_items = (level-i+1)*(level-i)//2
            # print("layer_items", layer_items)
            for j in range(0, level-i-1) :
                for k in range(0, level-i-j-1) :
                    p += 1
                    e0 = p
                    e1 = p+1
                    e2 = p+level-i-j
                    e3 = p+layer_items-j
                    print("first", e0-1, e3-1, e2-1, e1-1)

                    i0.append(e0-1)
                    i1.append(e1-1)
                    i2.append(e2-1)
                    i3.append(e3-1)
                    category.append(1)

                    micro_tets.append([ dofs[e0-1], dofs[e1-1], dofs[e2-1], dofs[e3-1] ])
                p += 1
            p += 1
            
        # print(micro_tets)

        # gather phase (can optimize out)
        # print("dofs", dofs)
        e0 = micro_tets[-1]
        p0 = np.array([x_coords[e0[0]], y_coords[e0[0]], z_coords[e0[0]]])
        p1 = np.array([x_coords[e0[1]], y_coords[e0[1]], z_coords[e0[1]]])
        p2 = np.array([x_coords[e0[2]], y_coords[e0[2]], z_coords[e0[2]]])
        p3 = np.array([x_coords[e0[3]], y_coords[e0[3]], z_coords[e0[3]]])
        # .. also gather coeffs from f

        # print("indices:", e0[0], e0[1], e0[2], e0[3])
        # print("positions:", p0, p1, p2, p3)
        
        # quadrature
        upright_A = compute_A(p0, p1, p2, p3, category=0)
        local_M = computeFn(upright_A, category=0)
        # scatter phase
        for i in range(0, 4) :
            for j in range(0, 4) :
                for e in micro_tets :
                    vecY[e[j]] += local_M[i, j] * vecX[e[i]]
        
        # Second subtet
        micro_tets = list()
        p = 0
        for i in range(0, level-1) :
            for j in range(0, level-i-1) :
                p += 1
                for k in range(1, level-i-j-1) :
                    p += 1
                    layer_items = (level-i)*(level-i-1)//2
                    e0 = p
                    e1 = p-i-j-1+layer_items+level
                    e2 = p-i-j+layer_items+level
                    e3 = p+level-i-j-1+layer_items+level-i-j-1
                    print("second", e0-1, e3-1, e2-1, e1-1)

                    i0.append(e0-1)
                    i1.append(e1-1)
                    i2.append(e2-1)
                    i3.append(e3-1)
                    category.append(2)

                    micro_tets.append([ dofs[e0-1], dofs[e1-1], dofs[e2-1], dofs[e3-1] ])
                p += 1
            p += 1
        
        # gather phase (can optimize out)
        # print(dofs)
        e1 = micro_tets[-1]
        p0 = np.array([x_coords[e1[0]], y_coords[e1[0]], z_coords[e1[0]]])
        p1 = np.array([x_coords[e1[1]], y_coords[e1[1]], z_coords[e1[1]]])
        p2 = np.array([x_coords[e1[2]], y_coords[e1[2]], z_coords[e1[2]]])
        p3 = np.array([x_coords[e1[3]], y_coords[e1[3]], z_coords[e1[3]]])
        # .. also gather coeffs from f
        
        # quadrature
        inverted_A = compute_A(p0, p1, p2, p3, category=1)
        local_M = computeFn(inverted_A, category=1)
        # scatter phase
        for i in range(0, 4) :
            for j in range(0, 4) :
                for e in micro_tets :
                    vecY[e[j]] += local_M[i, j] * vecX[e[i]]
        
        # Third subtet
        p = 0
        micro_tets = list()

        for i in range(0, level-1) :
            for j in range(0, level-i-1) :
                p += 1
                for k in range(1, level-i-j-1) :
                    p += 1
                    layer_items = (level-i)*(level-i-1)//2
                    e0 = p
                    e1 = p+level-i-j
                    e3 = p+level-i-j+layer_items
                    e2 = p+level-i-j-1+layer_items+level-i-j-1
#                     assert(tetrahedron_volume(np.array([x[e0-1], y[e0-1], z[e0-1]]),
#                                             np.array([x[e1-1], y[e1-1], z[e1-1]]),
#                                             np.array([x[e2-1], y[e2-1], z[e2-1]]),
#                                             np.array([x[e3-1], y[e3-1], z[e3-1]])) > 0)
                    print("third", e0-1, e3-1, e2-1, e1-1)
                    micro_tets.append([ dofs[e0-1], dofs[e1-1], dofs[e2-1], dofs[e3-1] ])

                    i0.append(e0-1)
                    i1.append(e1-1)
                    i2.append(e2-1)
                    i3.append(e3-1)
                    category.append(3)

                p += 1
            p += 1
        
        # gather phase (can optimize out)
        # print(dofs)
        e2 = micro_tets[-1]
        p0 = np.array([x_coords[e2[0]], y_coords[e2[0]], z_coords[e2[0]]])
        p1 = np.array([x_coords[e2[1]], y_coords[e2[1]], z_coords[e2[1]]])
        p2 = np.array([x_coords[e2[2]], y_coords[e2[2]], z_coords[e2[2]]])
        p3 = np.array([x_coords[e2[3]], y_coords[e2[3]], z_coords[e2[3]]])
        # .. also gather coeffs from f
        
        # quadrature
        third_A = compute_A(p0, p1, p2, p3, category=2)
        local_M = computeFn(third_A, category=2)
        # scatter phase
        for i in range(0, 4) :
            for j in range(0, 4) :
                for e in micro_tets :
                    vecY[e[j]] += local_M[i, j] * vecX[e[i]]
                    
        # Fourth subtet
        micro_tets = list()
        p = 0
        for i in range(0, level-1) :
            for j in range(0, level-i-1) :
                p += 1
                for k in range(1, level-i-j-1) :
                    p += 1
                    layer_items = (level-i)*(level-i-1)//2
                    e0 = p
                    e1 = p+level-i-j-1
                    e2 = p-i-j-1+layer_items+level
                    e3 = p+level-i-j-1+layer_items+level-i-j-1
                    micro_tets.append([ dofs[e0-1], dofs[e1-1], dofs[e2-1], dofs[e3-1] ])
                    print("forth", e0-1, e3-1, e2-1, e1-1)

                    i0.append(e0-1)
                    i1.append(e1-1)
                    i2.append(e2-1)
                    i3.append(e3-1)
                    category.append(4)

                p += 1
            p += 1
        
        # gather phase (can optimize out)
        # print(dofs)
        e3 = micro_tets[-1]
        p0 = np.array([x_coords[e3[0]], y_coords[e3[0]], z_coords[e3[0]]])
        p1 = np.array([x_coords[e3[1]], y_coords[e3[1]], z_coords[e3[1]]])
        p2 = np.array([x_coords[e3[2]], y_coords[e3[2]], z_coords[e3[2]]])
        p3 = np.array([x_coords[e3[3]], y_coords[e3[3]], z_coords[e3[3]]])
        # .. also gather coeffs from f

        # quadrature
        fourth_A = compute_A(p0, p1, p2, p3, category=3)
        local_M = computeFn(fourth_A, category=3)
        # scatter phase
        for i in range(0, 4) :
            for j in range(0, 4) :
                for e in micro_tets :
                    vecY[e[j]] += local_M[i, j] * vecX[e[i]]

        # Fifth subtet
        micro_tets = list()
        p = 0
        for i in range(1, level-1) :
            p = p + level - i + 1
            layer_items = (level-i)*(level-i-1)//2
            for j in range(0, level-i-1) :
                p += 1
                for k in range(1, level-i-j-1) :
                    p += 1
                    e0 = p
                    e1 = p+layer_items+level-i
                    e2 = p+layer_items+level-i-j+level-i
                    e3 = p+layer_items+level-i-j+level-i-1
                    
                    print("fifth", e0-1, e2-1, e1-1, e3-1)

                    i0.append(e0-1)
                    i1.append(e1-1)
                    i2.append(e2-1)
                    i3.append(e3-1)
                    category.append(5)

                    micro_tets.append([ dofs[e0-1], dofs[e1-1], dofs[e2-1], dofs[e3-1] ])
                p += 1
            p += 1
        
        # gather phase (can optimize out)
        # print(dofs)
        e4 = micro_tets[-1]
        p0 = np.array([x_coords[e4[0]], y_coords[e4[0]], z_coords[e4[0]]])
        p1 = np.array([x_coords[e4[1]], y_coords[e4[1]], z_coords[e4[1]]])
        p2 = np.array([x_coords[e4[2]], y_coords[e4[2]], z_coords[e4[2]]])
        p3 = np.array([x_coords[e4[3]], y_coords[e4[3]], z_coords[e4[3]]])
        # .. also gather coeffs from f
        assert(tetrahedron_volume(p0, p1, p2, p3) > 0)
        # quadrature
        fifth_A = compute_A(p0, p1, p2, p3, category=4)
        local_M = computeFn(fifth_A, category=4)
        # scatter phase
        for i in range(0, 4) :
            for j in range(0, 4) :
                for e in micro_tets :
                    vecY[e[j]] += local_M[i, j] * vecX[e[i]]
                    
        # Sixth subtet
        p = 0
        micro_tets = list()
        for i in range(0, level-1) :
            for j in range(0, level-i-1) :
                p += 1
                for k in range(1, level-i-j-1) :
                    p += 1
                    layer_items = (level-i)*(level-i-1)//2
                    e0 = p
                    e1 = p+level-i-j-1
                    e2 = p+level-i-j-1+layer_items+level-i-j-1
                    e3 = p+level-i-j

                    i0.append(e0-1)
                    i1.append(e1-1)
                    i2.append(e2-1)
                    i3.append(e3-1)
                    category.append(6)

                    print("sixth", e0-1, e2-1, e1-1, e3-1)

                    micro_tets.append([ dofs[e0-1], dofs[e1-1], dofs[e2-1], dofs[e3-1] ])
                p += 1
            p += 1
        
        # gather phase (can optimize out)
        # print(dofs)
        e5 = micro_tets[-1]
        p0 = np.array([x_coords[e5[0]], y_coords[e5[0]], z_coords[e5[0]]])
        p1 = np.array([x_coords[e5[1]], y_coords[e5[1]], z_coords[e5[1]]])
        p2 = np.array([x_coords[e5[2]], y_coords[e5[2]], z_coords[e5[2]]])
        p3 = np.array([x_coords[e5[3]], y_coords[e5[3]], z_coords[e5[3]]])
        # .. also gather coeffs from f

        
        # quadrature
        sixth_A = compute_A(p0, p1, p2, p3, category=5)
        # compute_Lapl
        local_M = computeFn(sixth_A, category=5)
        # scatter phase
        for i in range(0, 4) :
            for j in range(0, 4) :
                for e in micro_tets :
                    vecY[e[j]] += local_M[i, j] * vecX[e[i]]

    np.array(i0).astype(np.int32).tofile('i0.raw')
    np.array(i1).astype(np.int32).tofile('i1.raw')
    np.array(i2).astype(np.int32).tofile('i2.raw')
    np.array(i3).astype(np.int32).tofile('i3.raw')
    np.array(category).astype(np.float64).tofile('category.raw')

    return vecY

# 1 

def apply_macro_kernel(in_list, tetra_level, nodes, x_coords, y_coords, z_coords, computeFn, vecX) :
    vecY = assemble_macro_elem(in_list, tetra_level, nodes, x_coords, y_coords, z_coords, computeFn, vecX)
    return vecY


def set_boundary_conditions(nodes) :
    # dirichlet_nodes = np.array([0])
    # dirichlet_value = np.array([1])
    # should give all 1s

    # second cond
    dirichlet_nodes = np.array([0, nodes-1])
    dirichlet_value = np.array([1, 0])

    rhs = np.zeros(nodes)
    x = np.zeros(nodes)

    for i in range(0, len(dirichlet_nodes)) :
        idx = dirichlet_nodes[i]
        rhs[idx] = dirichlet_value[i]
        x[idx] = dirichlet_value[i]

    return rhs, x, dirichlet_nodes

# rhs is the right hand side of the system of equation
# we set the boundary conditions here
def residual(in_list, tetra_level, nodes, x_coords, y_coords, z_coords, compute_Lapl, dirichlet_nodes, rhs, x) :
    
    Ax = apply_macro_kernel(in_list, tetra_level, nodes, x_coords, y_coords, z_coords, compute_Lapl, x)
    #print("Ax", Ax)
    for dirichlet_node in dirichlet_nodes :
        Ax[dirichlet_node] = x[dirichlet_node]
    # print(Ax.shape, rhs.shape)
#     assert(Ax.shape[1] == 1)
#     assert(rhs.shape[1] == 1)
    r = rhs - Ax.flatten()
    return r

def compute_Lapl(J, category=0) :
    A = np.zeros((4, 4))
    
    # print(J)
    det_J = np.linalg.det(J)
    assert det_J > 0, "Determinant should be positive"
    
    J_inv = np.linalg.inv(J)
    J_inv_trans = np.transpose(J_inv)
    
    grad_ref_phi = list()
    grad_ref_phi.append(np.array([-1, -1, -1]))
    grad_ref_phi.append(np.array([1, 0, 0]))
    grad_ref_phi.append(np.array([0, 1, 0]))
    grad_ref_phi.append(np.array([0, 0, 1]))

#     # print("J_inv_trans", J_inv_trans)
#     # print("J_inv_trans * grad_ref_phi[i]", np.matmul(J_inv_trans, grad_ref_phi[0]))
    
    grad_phi = list()
    for i in range(4) :
        grad_phi.append(np.matmul(J_inv_trans, grad_ref_phi[i]))
    
    for i in range(4) :
        for j in range(4) :
            # print(i, j, grad_phi[i], np.dot(grad_phi[i] , grad_phi[j]) * np.linalg.det(J)/2)
            A[i, j] = np.dot(grad_phi[i] , grad_phi[j]) * det_J / 6

    if not lapl_shown[category] :
        print("Laplacian", category)
        print(A)
        lapl_shown[category] = True

    check_laplacian_matrix(A)

    return A

def generate_coords(tetra_level) :
    x_coords = list()
    y_coords = list()
    z_coords = list()

    for k in range(0, tetra_level + 1):
        for j in range(0, tetra_level - k + 1):
            for i in range(0, tetra_level - j - k + 1):
                x_coords.append(1 * i / tetra_level)
                y_coords.append(1 * j / tetra_level)
                z_coords.append(1 * k / tetra_level)

    # print(x_coords)
    # print(y_coords)
    # print(z_coords)

    np.array(x_coords).astype(np.float32).tofile('x.raw')
    np.array(y_coords).astype(np.float32).tofile('y.raw')
    np.array(z_coords).astype(np.float32).tofile('z.raw')

    return x_coords, y_coords, z_coords

def main() :

    # compute the number of elems

    tetra_level = 8

    x_coords, y_coords, z_coords = generate_coords(tetra_level)

    in_list = create_macro_tet4_mesh(tetra_level=tetra_level)
    nodes = compute_nodes_number(tetra_level=tetra_level)
    rhs, x, dirichlet_nodes = set_boundary_conditions(nodes)

    print(len(x_coords), len(x))
    assert(len(x_coords) == len(x))

    max_iters = 1
    for i in range(0, max_iters) : 
        r = residual(in_list, tetra_level, nodes, x_coords, y_coords, z_coords, compute_Lapl, dirichlet_nodes, rhs, x)
        #print("residual",r)
        norm_r = math.sqrt(np.dot(r, r))
        # gamma can be too high, we can make it smaller
        gamma = 7 * 1e-1
        print(norm_r)
        x = x + gamma * r
        if norm_r < 1e-6 :
            break

    print(x)

    np.array(x).astype(np.float64).tofile('solution.raw')

# Example usage
if __name__ == "__main__":
    main()
    os.chdir("/Users/bolema/Documents/sfem/")
    os.system("source venv/bin/activate && cd python/sfem/mesh/ && python3 raw_to_db.py /Users/bolema/Documents/hpcfem/python /Users/bolema/Documents/hpcfem/mesh.vtk -c /Users/bolema/Documents/hpcfem/python/category.raw -p /Users/bolema/Documents/hpcfem/python/solution.raw")

# nodes = 2**(level+1) * macro_tets
# what about shared faces?

# x_coords = [0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 0.0, 0.25, 0.5, 0.0, 0.25, 0.0, 0.0, 0.25, 0.5, 0.75, 0.0, 0.25, 0.5, 0.0, 0.25, 0.0, 0.0, 0.25, 0.5, 0.0, 0.25, 0.0, 0.0, 0.25, 0.0, 0.0]
# y_coords = [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.75, 0.75, 1.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.5, 0.5, 0.75, 0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.0, 0.0, 0.25, 0.0]
# z_coords = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 1.0]

# print(nodes)
# nodes = nodes * macro_tets