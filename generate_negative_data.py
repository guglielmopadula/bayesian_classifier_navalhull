#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: cyberguli
"""

import scipy
import numpy as np
np.random.seed(0)
import meshio
from tqdm import trange
from sklearn.decomposition import PCA
from skdim.id import TwoNN
from scipy.stats import truncnorm
from scipy.stats import multivariate_normal

def linear_distribution(N):
    return -2/(N*N+N)*np.arange(1,N+1)+2/N
    

def getinfo(stl,flag):
    mesh=meshio.read(stl)
    mesh.points[abs(mesh.points)<10e-05]=0
    points_old=mesh.points.astype(np.float32)
    points=points_old[np.logical_and(points_old[:,2]>0,points_old[:,0]>0)]
    if flag==True:
        newmesh_indices_global=np.arange(len(mesh.points))[np.logical_and(points_old[:,2]>0,points_old[:,0]>0)].tolist()
        triangles=mesh.cells_dict['triangle'].astype(np.int64)
        newtriangles=[]
        for T in triangles:
            if T[0] in newmesh_indices_global and T[1] in newmesh_indices_global and T[2] in newmesh_indices_global:
                newtriangles.append([newmesh_indices_global.index(T[0]),newmesh_indices_global.index(T[1]),newmesh_indices_global.index(T[2])])
        #newmesh_indices_global_zero=np.arange(len(mesh.points))[np.logical_and(points_old[:,2]>=0,points_old[:,0]>=0)].tolist()
        tmp=triangles[np.in1d(triangles,np.array(newmesh_indices_global)).reshape(-1,3).sum(axis=1).astype(bool)]
        newmesh_indices_global_zero=np.unique(tmp.reshape(-1)).tolist()
        points_zero=points_old[newmesh_indices_global_zero]
        newtriangles_zero=[]
        for T in triangles:
            if T[0] in newmesh_indices_global_zero and T[1] in newmesh_indices_global_zero and T[2] in newmesh_indices_global_zero:
                newtriangles_zero.append([newmesh_indices_global_zero.index(T[0]),newmesh_indices_global_zero.index(T[1]),newmesh_indices_global_zero.index(T[2])])
        newmesh_indices_local=np.arange(len(points_zero))[np.logical_and(points_zero[:,2]>0,points_zero[:,0]>0)].tolist()
        newtriangles_local_3=[]
        newtriangles_local_2=[]
        newtriangles_local_1=[]
        edge_matrix=np.zeros([np.max(newtriangles_zero)+1,np.max(newtriangles_zero)+1])
        vertices_face=[set({}) for i in range(len(newmesh_indices_local))]
        for T in newtriangles_zero:
            if sum((int(T[0] in newmesh_indices_local),int(T[1] in newmesh_indices_local),int(T[2] in newmesh_indices_local)))==3:
                newtriangles_local_3.append([T[0],T[1],T[2]])
            if sum((int(T[0] in newmesh_indices_local),int(T[1] in newmesh_indices_local),int(T[2] in newmesh_indices_local)))==2:
                newtriangles_local_2.append([T[0],T[1],T[2]])
            if sum((int(T[0] in newmesh_indices_local),int(T[1] in newmesh_indices_local),int(T[2] in newmesh_indices_local)))==1:
                newtriangles_local_1.append([T[0],T[1],T[2]])
        
        for i in range(len(newtriangles_zero)):
            T=newtriangles_zero[i]
            if T[0] in newmesh_indices_local:
                edge_matrix[T[0],T[1]]=1
                edge_matrix[T[0],T[2]]=1
                vertices_face[newmesh_indices_local.index(T[0])].add(i)
            else:
                edge_matrix[T[0],T[0]]=1
                
            if T[1] in newmesh_indices_local:
                edge_matrix[T[1],T[2]]=1
                edge_matrix[T[1],T[0]]=1
                vertices_face[newmesh_indices_local.index(T[1])].add(i)
            else:
                edge_matrix[T[1],[T[1]]]=1
                
                
            if T[2] in newmesh_indices_local:
                edge_matrix[T[2],T[0]]=1
                edge_matrix[T[2],T[1]]=1
                vertices_face[newmesh_indices_local.index(T[2])].add(i)
            else:
                edge_matrix[T[2],[T[2]]]=1
        vertices_face=[list(t) for t in vertices_face]

    else:
        triangles=0
        newtriangles=0
        newmesh_indices_local=0
        newtriangles_zero=0
        newtriangles_local_1=0
        newtriangles_local_2=0
        newtriangles_local_3=0
        vertices_face=0
        edge_matrix=0
        
    return points,points_zero,points_old,newmesh_indices_local,triangles,newtriangles_zero,newtriangles_local_1,newtriangles_local_2,newtriangles_local_3,newmesh_indices_global_zero,edge_matrix,vertices_face


points,points_zero,points_old,newmesh_indices_local,triangles,newtriangles_zero,newtriangles_local_1,newtriangles_local_2,newtriangles_local_3,newmesh_indices_global_zero,edge_matrix,vertices_face=getinfo("positive_data/hull_0.stl",True)


temp1=points_zero[np.logical_and(points_zero[:,2]>0,points_zero[:,0]>0)]
NUM_SAMPLES=600
alls=np.zeros([NUM_SAMPLES,2572,3])
points_list=np.arange(points_old.shape[0])[np.logical_and(points_old[:,2]>0,points_old[:,0]>0)]
features_index=np.load("features_index.npy")


for i in trange(NUM_SAMPLES):
    tot_arr=meshio.read("positive_data/hull_{}.stl".format(i)).points
    points=tot_arr[points_list].reshape(-1)
    points2=points[features_index]
    a=np.random.choice(len(points2),1,p=linear_distribution(len(points2)))
    v=np.random.choice(len(points2), a, replace=False)
    #matrix=np.random.rand(3*a[0],3*a[0])
    #matrix=0.5*(matrix+matrix.T)+3*a[0]*np.eye(3*a[0])
    #matrix=matrix/np.diag(matrix)
    #points[v]+=0.5*1/np.sqrt(a[0])*multivariate_normal(mean=None, cov=matrix, allow_singular=False).rvs().reshape(-1,3)
    points2[v]+=0.5*1/np.sqrt(a[0])*np.random.rand(a[0])
    points[features_index]=points2
    points=points.reshape(-1,3)
    tot_arr[points_list]=points
    meshio.write_points_cells("./negative_data/hull_{}.stl".format(i), tot_arr, [("triangle", triangles)])
    
