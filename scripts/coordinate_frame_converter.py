import numpy as np
import cv2

class CoordinateFrameConverter:
    def __init__(self,T_cs,mesh_props,K) -> None:

        self.bbox = mesh_props["bbox"]
        self.extents = mesh_props["extents"]
        self._K = K

        # Get shelf corners in C frame
        bounding_box =self.bbox
        corners_S = self.get_box_corners(bounding_box)
        corners_C = []
        for p_S in corners_S:
            p_C =(T_cs @ p_S)[0:3]
            corners_C.append(p_C)

        # Get transform from C-frame to E-frame
        self.T_ce = self.get_T_ce(corners_C)

    def get_box_corners(self,bbox):
        min_xyz = bbox.min(axis=0)
        xmin, ymin, zmin = min_xyz
        max_xyz = bbox.max(axis=0)
        xmax, ymax, zmax = max_xyz

        box_corners_S = []

        for x in [xmin,xmax]:
            for y in [ymin,ymax]:
                for z in [zmin,zmax]:
                    p_S = np.array([x,y,z,1])
                    box_corners_S.append(p_S)
        
        return box_corners_S

    def get_L(self,corners_C):
        return sorted(corners_C,key=lambda pt: pt[0])[0:4]

    def get_R(self,corners_C):
        return sorted(corners_C,key=lambda pt: -pt[0])[0:4]

    def get_T_ce(self,corners_C):
        # ID CORNERS ##################################################

        # Get left corners
        L_C = self.get_L(corners_C)

        # Id H,J by distance to camera
        L_C = sorted(L_C,key=lambda vec: np.linalg.norm(vec))
        H = L_C[0]
        J = L_C[3]

        # Id A,D by distance to H,J
        L1 = L_C[1]
        L2 = L_C[2]

        d_L1H = np.linalg.norm(H-L1)
        d_L2H = np.linalg.norm(H-L2)

        if d_L1H < d_L2H:
            A = L1
            D = L2
        else:
            A = L2
            D = L1

        # Get right corners
        R_C = self.get_R(corners_C)

        # Id G,F by distance to camera
        R_C = sorted(R_C,key=lambda vec: np.linalg.norm(vec))
        G = R_C[0]
        F = R_C[3]

        # Id I,E by distance to G,G
        R1 = R_C[1]
        R2 = R_C[2]

        d_R1G = np.linalg.norm(G-R1)
        d_R2G = np.linalg.norm(G-R2)
        if d_R1G < d_R2G:
            I = R1
            E = R2
        else:
            I = R2
            E = R1


        x = F - E
        y = D - E
        z = G - E

        xn = x / np.linalg.norm(x)
        yn = y / np.linalg.norm(y)
        zn = z / np.linalg.norm(z)

        T_ce = np.zeros((4,4))
        T_ce[0:3,0] = xn
        T_ce[0:3,1] = yn
        T_ce[0:3,2] = zn
        T_ce[0:3,3] = E
        T_ce[3,3] = 1

        return T_ce
