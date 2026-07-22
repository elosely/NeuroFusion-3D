import numpy as np
import plotly.graph_objects as go
from skimage import measure

class Medical3DMeshBuilder:
    """
    Generates Multi-Color 3D Surface Meshes separating Healthy Brain, 
    Tumor Edema/Boundaries, and Active Tumor Core with distinct thresholds.
    """

    @staticmethod
    def create_3d_volume_mesh(volume: np.ndarray, core_mask: np.ndarray, edema_mask: np.ndarray, step_size: int = 2) -> go.Figure:
        fig = go.Figure()

        # 1. BRAIN SHELL (Transparent Slate Gray)
        brain_mask = (volume > 0.15) & (volume <= 0.65)
        if np.sum(brain_mask) > 0:
            vol_sub = brain_mask[::step_size, ::step_size, ::step_size]
            if np.sum(vol_sub) > 0:
                verts_b, faces_b, _, _ = measure.marching_cubes(vol_sub, level=0.5)
                
                fig.add_trace(go.Mesh3d(
                    x=verts_b[:, 0] * step_size,
                    y=verts_b[:, 1] * step_size,
                    z=verts_b[:, 2] * step_size,
                    i=faces_b[:, 0],
                    j=faces_b[:, 1],
                    k=faces_b[:, 2],
                    color='#CBD5E0',
                    opacity=0.10,
                    name='Brain Shell (Gray Matter)',
                    hoverinfo='name'
                ))

        # 2. TUMOR EDEMA / BOUNDARY (Bright Amber / Yellow)
        if np.sum(edema_mask) > 0:
            edema_sub = edema_mask[::step_size, ::step_size, ::step_size]
            if np.sum(edema_sub) > 0:
                verts_e, faces_e, _, _ = measure.marching_cubes(edema_sub, level=0.5)
                fig.add_trace(go.Mesh3d(
                    x=verts_e[:, 0] * step_size,
                    y=verts_e[:, 1] * step_size,
                    z=verts_e[:, 2] * step_size,
                    i=faces_e[:, 0],
                    j=faces_e[:, 1],
                    k=faces_e[:, 2],
                    color='#ECC94B',  # Amber Yellow
                    opacity=0.45,
                    name='Peritumoral Edema / Boundary',
                    hoverinfo='name'
                ))

        # 3. ACTIVE TUMOR CORE (Solid Deep Red)
        if np.sum(core_mask) > 0:
            core_sub = core_mask[::step_size, ::step_size, ::step_size]
            if np.sum(core_sub) > 0:
                verts_c, faces_c, _, _ = measure.marching_cubes(core_sub, level=0.5)
                fig.add_trace(go.Mesh3d(
                    x=verts_c[:, 0] * step_size,
                    y=verts_c[:, 1] * step_size,
                    z=verts_c[:, 2] * step_size,
                    i=faces_c[:, 0],
                    j=faces_c[:, 1],
                    k=faces_c[:, 2],
                    color='#E53E3E',  # Deep Red
                    opacity=0.90,
                    name='Active Tumor Core',
                    hoverinfo='name'
                ))

        # Layout Settings
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data',
                camera=dict(
                    eye=dict(x=1.6, y=1.6, z=1.3)
                )
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            height=550,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.7)"
            )
        )

        return fig