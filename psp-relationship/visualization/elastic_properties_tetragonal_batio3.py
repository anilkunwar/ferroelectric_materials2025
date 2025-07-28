import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize session state for input synchronization
if 'matrix_input' not in st.session_state:
    st.session_state.matrix_input = """260 108 77 0 0 0
108 260 77 0 0 0
77 77 89 0 0 0
0 0 0 31 0 0
0 0 0 0 31 0
0 0 0 0 0 116"""
if 'row_inputs' not in st.session_state:
    st.session_state.row_inputs = [
        "260 108 77 0 0 0",
        "108 260 77 0 0 0",
        "77 77 89 0 0 0",
        "0 0 0 31 0 0",
        "0 0 0 0 31 0",
        "0 0 0 0 0 116"
    ]
if 'material_name' not in st.session_state:
    st.session_state.material_name = "BaTiO3 (Tetragonal)"

# Streamlit app title
st.title(f"3D Visualization of Elastic Properties for {st.session_state.material_name}")

# Material name input
st.sidebar.subheader("Material Name")
material_name = st.sidebar.text_input("Enter material name manually", value=st.session_state.material_name, key="material_name")

# LaTeX description of stiffness matrix and formulas
st.subheader("Stiffness Matrix")
st.latex(r"""
\mathbf{C} = \begin{pmatrix}
C_{11} & C_{12} & C_{13} & C_{14} & C_{15} & C_{16} \\
C_{12} & C_{22} & C_{23} & C_{24} & C_{25} & C_{26} \\
C_{13} & C_{23} & C_{33} & C_{34} & C_{35} & C_{36} \\
C_{14} & C_{24} & C_{34} & C_{44} & C_{45} & C_{46} \\
C_{15} & C_{25} & C_{35} & C_{45} & C_{55} & C_{56} \\
C_{16} & C_{26} & C_{36} & C_{46} & C_{56} & C_{66}
\end{pmatrix}
""")
st.subheader("The following visualization is applicable to all crystal symmetries for 3D materials, 
including cubic, tetragonal, orthorhombic, and lower symmetries like
monoclinic and triclinic, when provided with a valid 6x6 stiffness matrix in Voigt notation.")

# Input for stiffness matrix
st.sidebar.subheader(f"Stiffness Matrix Input (6x6, GPa) for {material_name} ")
#st.sidebar.subheader(f"Stiffness Matrix of {material_name}")
st.sidebar.markdown("""
Enter the 6x6 stiffness matrix (in GPa) below. You can paste the full matrix in the textarea (rows separated by newlines, values by spaces) or enter each row individually. Changes in one will update the other.
""")
def update_rows_from_matrix():
    try:
        rows = st.session_state.matrix_input.strip().split('\n')
        if len(rows) == 6:
            for i, row in enumerate(rows):
                if len(row.split()) == 6:
                    st.session_state.row_inputs[i] = row.strip()
    except:
        pass

def update_matrix_from_rows():
    try:
        st.session_state.matrix_input = '\n'.join([st.session_state[f"row_{i}"] for i in range(6)])
    except:
        pass

matrix_input = st.sidebar.text_area("Paste full 6x6 matrix (rows separated by newlines)", 
                            value=st.session_state.matrix_input, 
                            key="matrix_input", 
                            on_change=update_rows_from_matrix)

st.sidebar.markdown("**Or enter rows individually:**")
rows = []
for i in range(6):
    row = st.sidebar.text_input(f"Row {i+1}", value=st.session_state.row_inputs[i], 
                        key=f"row_{i}", on_change=update_matrix_from_rows)
    rows.append(row)

# Parse input
try:
    if matrix_input.strip():
        C = np.array([[float(x) for x in row.split()] for row in matrix_input.strip().split('\n')])
    else:
        C = np.array([[float(x) for x in row.split()] for row in rows])
    if C.shape != (6, 6):
        st.error("Matrix must be 6x6. Please check the input.")
        st.stop()
except ValueError:
    st.error("Invalid input. Please enter numerical values separated by spaces.")
    st.stop()

# Compute compliance matrix
try:
    S = inv(C)
except np.linalg.LinAlgError:
    st.error("Stiffness matrix is singular or invalid. Please check the input.")
    st.stop()

# Colormap selection
colormaps = [
    'Viridis', 'Cividis', 'Plasma', 'Inferno', 'Magma', 'Hot', 'Cool', 'Rainbow', 'Jet',
    'Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 'Greys', 'YlOrRd', 'YlOrBr', 'YlGn',
    'RdBu', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
    'Coolwarm', 'Bwr', 'Seismic', 'Twilight', 'Twilight_shifted', 'Hsv', 'Flag', 'Prism',
    'Ocean', 'Gist_earth', 'Terrain', 'Gist_stern', 'Gnuplot', 'Gnuplot2', 'CMRmap',
    'Cubehelix', 'Brg', 'Gist_rainbow', 'Nipy_spectral', 'Gist_ncar', 'Pink', 'Spring',
    'Summer', 'Autumn', 'Winter', 'Bone', 'Copper'
]
colormap_mapping = {
    'Viridis': 'viridis', 'Cividis': 'cividis', 'Plasma': 'plasma', 'Inferno': 'inferno',
    'Magma': 'magma', 'Hot': 'hot', 'Cool': 'cool', 'Rainbow': 'rainbow', 'Jet': 'jet',
    'Blues': 'Blues', 'Greens': 'Greens', 'Reds': 'Reds', 'Purples': 'Purples',
    'Oranges': 'Oranges', 'Greys': 'Greys', 'YlOrRd': 'YlOrRd', 'YlOrBr': 'YlOrBr',
    'YlGn': 'YlGn', 'RdBu': 'RdBu', 'PiYG': 'PiYG', 'PRGn': 'PRGn', 'BrBG': 'BrBG',
    'PuOr': 'PuOr', 'RdGy': 'RdGy', 'RdYlBu': 'RdYlBu', 'RdYlGn': 'RdYlGn',
    'Spectral': 'Spectral', 'Coolwarm': 'coolwarm', 'Bwr': 'bwr', 'Seismic': 'seismic',
    'Twilight': 'twilight', 'Twilight_shifted': 'twilight_shifted', 'Hsv': 'hsv',
    'Flag': 'flag', 'Prism': 'prism', 'Ocean': 'ocean', 'Gist_earth': 'gist_earth',
    'Terrain': 'terrain', 'Gist_stern': 'gist_stern', 'Gnuplot': 'gnuplot',
    'Gnuplot2': 'gnuplot2', 'CMRmap': 'CMRmap', 'Cubehelix': 'cubehelix', 'Brg': 'brg',
    'Gist_rainbow': 'gist_rainbow', 'Nipy_spectral': 'nipy_spectral',
    'Gist_ncar': 'gist_ncar', 'Pink': 'pink', 'Spring': 'spring', 'Summer': 'summer',
    'Autumn': 'autumn', 'Winter': 'winter', 'Bone': 'bone', 'Copper': 'copper'
}
st.sidebar.subheader("Select Colormaps")
selected_colormap_E = st.sidebar.selectbox("Colormap for Young's Modulus", colormaps, index=0, key="colormap_E")
selected_colormap_nu = st.sidebar.selectbox("Colormap for Poisson's Ratio", colormaps, index=19, key="colormap_nu")
selected_colormap_G = st.sidebar.selectbox("Colormap for Shear Modulus", colormaps, index=0, key="colormap_G")
selected_colormap_beta = st.sidebar.selectbox("Colormap for Linear Compressibility", colormaps, index=19, key="colormap_beta")

# Function to compute elastic properties
def compute_properties(direction, S):
    l, m, n = direction / np.linalg.norm(direction)
    a = np.array([l**2, m**2, n**2, 2*m*n, 2*n*l, 2*l*m])
    E = 1 / (a.T @ S @ a)  # Young's modulus in GPa

    # Poisson's ratio
    if abs(l) > 1e-6 or abs(m) > 1e-6:
        b1 = np.array([-m, l, 0])
        b2 = np.cross(direction, b1)
    else:
        b1 = np.array([0, -n, m])
        b2 = np.cross(direction, b1)
    b1 = b1 / np.linalg.norm(b1)
    b2 = b2 / np.linalg.norm(b2)
    b1_voigt = np.array([b1[0]**2, b1[1]**2, b1[2]**2, 2*b1[1]*b1[2], 2*b1[2]*b1[0], 2*b1[0]*b1[1]])
    b2_voigt = np.array([b2[0]**2, b2[1]**2, b2[2]**2, 2*b2[1]*b2[2], 2*b2[2]*b2[0], 2*b2[0]*b2[1]])
    nu1 = -(a.T @ S @ b1_voigt) / (a.T @ S @ a)
    nu2 = -(a.T @ S @ b2_voigt) / (a.T @ S @ a)
    nu = (nu1 + nu2) / 2

    # Shear modulus
    b1_shear = np.array([l*b1[0], m*b1[1], n*b1[2], m*b1[2] + b1[1]*n, n*b1[0] + b1[2]*l, l*b1[1] + b1[0]*m])
    b2_shear = np.array([l*b2[0], m*b2[1], n*b2[2], m*b2[2] + b2[1]*n, n*b2[0] + b2[2]*l, l*b2[1] + b2[0]*m])
    G1 = 1 / (4 * (b1_shear.T @ S @ b1_shear))
    G2 = 1 / (4 * (b2_shear.T @ S @ b2_shear))
    G = (G1 + G2) / 2  # Average for stability

    # Linear compressibility
    a_comp = np.array([l**2, m**2, n**2, 0, 0, 0])
    c = np.array([1, 1, 1, 0, 0, 0])
    beta = a_comp.T @ S @ c  # in GPa^-1

    return E, nu, G, beta

# Generate points for 3D plot
theta = np.linspace(0, np.pi, 50)
phi = np.linspace(0, 2 * np.pi, 50)
theta, phi = np.meshgrid(theta, phi)
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Compute elastic properties
E_values = np.zeros_like(x)
nu_values = np.zeros_like(x)
G_values = np.zeros_like(x)
beta_values = np.zeros_like(x)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        direction = np.array([x[i, j], y[i, j], z[i, j]])
        E, nu, G, beta = compute_properties(direction, S)
        E_values[i, j] = E
        nu_values[i, j] = nu
        G_values[i, j] = G
        beta_values[i, j] = beta

# Validate properties
nu_12 = -S[0,1] / S[0,0]
if not (0.2 <= nu_12 <= 0.5):  # Broader range for general materials
    st.warning(f"Poisson's ratio along [100] is {nu_12:.3f}, which is outside the typical range (0.2–0.5). Please verify the stiffness matrix.")
if np.max(nu_values) > 0.5 or np.min(nu_values) < -0.5:
    st.warning(f"Directional Poisson's ratio range [{np.min(nu_values):.3f}, {np.max(nu_values):.3f}] is unusual. Typical values are between -0.5 and 0.5.")
if np.min(G_values) < 0 or np.max(G_values) > 200:
    st.warning(f"Shear modulus range [{np.min(G_values):.2f}, {np.max(G_values):.2f}] GPa is unusual. Typical values are positive and below 200 GPa.")
if np.min(beta_values) < 0 or np.max(beta_values) > 0.1:
    st.warning(f"Linear compressibility range [{np.min(beta_values):.3f}, {np.max(beta_values):.3f}] GPa^-1 is unusual. Typical values are positive and below 0.1 GPa^-1.")

# Scale coordinates
E_max = np.max(E_values)
x_E = x * E_values / E_max
y_E = y * E_values / E_max
z_E = z * E_values / E_max
nu_max = np.max(np.abs(nu_values))
x_nu = x * np.abs(nu_values) / nu_max
y_nu = y * np.abs(nu_values) / nu_max
z_nu = z * np.abs(nu_values) / nu_max
G_max = np.max(G_values)
x_G = x * G_values / G_max
y_G = y * G_values / G_max
z_G = z * G_values / G_max
beta_max = np.max(np.abs(beta_values))
x_beta = x * np.abs(beta_values) / beta_max
y_beta = y * np.abs(beta_values) / beta_max
z_beta = z * np.abs(beta_values) / beta_max
x_sphere = x
y_sphere = y
z_sphere = z

# Create tabs for each property
tab1, tab2, tab3, tab4 = st.tabs(["Young's Modulus", "Poisson's Ratio", "Shear Modulus", "Linear Compressibility"])

# Young's Modulus Tab
with tab1:
    st.subheader(f"Young's Modulus for {material_name}")
    # Plotly
    fig_E = go.Figure()
    fig_E.add_trace(go.Surface(
        x=x_E, y=y_E, z=z_E, surfacecolor=E_values, colorscale=selected_colormap_E,
        colorbar=dict(title="Young's Modulus (GPa)"), showscale=True
    ))
    fig_E.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere, surfacecolor=np.ones_like(x_sphere),
        colorscale=[[0, 'rgba(128, 128, 128, 0.2)'], [1, 'rgba(128, 128, 128, 0.2)']], showscale=False
    ))
    max_E_idx = np.unravel_index(np.argmax(E_values), E_values.shape)
    min_E_idx = np.unravel_index(np.argmin(E_values), E_values.shape)
    max_E_point = [x_E[max_E_idx], y_E[max_E_idx], z_E[max_E_idx]]
    min_E_point = [x_E[min_E_idx], y_E[min_E_idx], z_E[min_E_idx]]
    fig_E.add_trace(go.Scatter3d(
        x=[max_E_point[0], min_E_point[0]], y=[max_E_point[1], min_E_point[1]], z=[max_E_point[2], min_E_point[2]],
        mode='markers+text', text=[f'Max: {E_values[max_E_idx]:.2f} GPa', f'Min: {E_values[min_E_idx]:.2f} GPa'],
        textposition='top center', marker=dict(size=5, color='red')
    ))
    fig_E.update_layout(
        title=f"Young's Modulus (GPa) for {material_name} (Plotly)",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode='cube',
                   xaxis=dict(range=[-1.2, 1.2]), yaxis=dict(range=[-1.2, 1.2]), zaxis=dict(range=[-1.2, 1.2])),
        width=600, height=600
    )
    st.plotly_chart(fig_E)
    
    # Matplotlib
    fig_mpl_E = plt.figure(figsize=(6, 6))
    ax_E = fig_mpl_E.add_subplot(111, projection='3d')
    ax_E.plot_surface(x_E, y_E, z_E, facecolors=plt.cm.get_cmap(colormap_mapping[selected_colormap_E])(E_values/E_max), alpha=0.8)
    ax_E.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.2)
    ax_E.scatter([max_E_point[0], min_E_point[0]], [max_E_point[1], min_E_point[1]], [max_E_point[2], min_E_point[2]], color='red', s=50)
    ax_E.text(max_E_point[0], max_E_point[1], max_E_point[2], f'Max: {E_values[max_E_idx]:.2f} GPa', size=10)
    ax_E.text(min_E_point[0], min_E_point[1], min_E_point[2], f'Min: {E_values[min_E_idx]:.2f} GPa', size=10)
    ax_E.set_xlabel('X')
    ax_E.set_ylabel('Y')
    ax_E.set_zlabel('Z')
    ax_E.set_title(f"Young's Modulus (GPa) for {material_name} (Matplotlib)")
    ax_E.set_box_aspect([1,1,1])
    ax_E.set_xlim([-1.2, 1.2])
    ax_E.set_ylim([-1.2, 1.2])
    ax_E.set_zlim([-1.2, 1.2])
    st.pyplot(fig_mpl_E)
    
    st.write(f"Young's Modulus (along [100]): {1/S[0,0]:.2f} GPa")
    st.write(f"Maximum Young's Modulus: {E_values[max_E_idx]:.2f} GPa")
    st.write(f"Minimum Young's Modulus: {E_values[min_E_idx]:.2f} GPa")

# Poisson's Ratio Tab
with tab2:
    st.subheader(f"Poisson's Ratio for {material_name}")
    # Plotly
    fig_nu = go.Figure()
    fig_nu.add_trace(go.Surface(
        x=x_nu, y=y_nu, z=z_nu, surfacecolor=nu_values, colorscale=selected_colormap_nu,
        colorbar=dict(title="Poisson's Ratio"), showscale=True
    ))
    fig_nu.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere, surfacecolor=np.ones_like(x_sphere),
        colorscale=[[0, 'rgba(128, 128, 128, 0.2)'], [1, 'rgba(128, 128, 128, 0.2)']], showscale=False
    ))
    max_nu_idx = np.unravel_index(np.argmax(nu_values), nu_values.shape)
    min_nu_idx = np.unravel_index(np.argmin(nu_values), nu_values.shape)
    max_nu_point = [x_nu[max_nu_idx], y_nu[max_nu_idx], z_nu[max_nu_idx]]
    min_nu_point = [x_nu[min_nu_idx], y_nu[min_nu_idx], z_nu[min_nu_idx]]
    fig_nu.add_trace(go.Scatter3d(
        x=[max_nu_point[0], min_nu_point[0]], y=[max_nu_point[1], min_nu_point[1]], z=[max_nu_point[2], min_nu_point[2]],
        mode='markers+text', text=[f'Max: {nu_values[max_nu_idx]:.3f}', f'Min: {nu_values[min_nu_idx]:.3f}'],
        textposition='top center', marker=dict(size=5, color='red')
    ))
    fig_nu.update_layout(
        title=f"Poisson's Ratio for {material_name} (Plotly)",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode='cube',
                   xaxis=dict(range=[-1.2, 1.2]), yaxis=dict(range=[-1.2, 1.2]), zaxis=dict(range=[-1.2, 1.2])),
        width=600, height=600
    )
    st.plotly_chart(fig_nu)
    
    # Matplotlib
    fig_mpl_nu = plt.figure(figsize=(6, 6))
    ax_nu = fig_mpl_nu.add_subplot(111, projection='3d')
    norm_nu = (nu_values - np.min(nu_values)) / (np.max(nu_values) - np.min(nu_values)) if np.max(nu_values) != np.min(nu_values) else nu_values
    ax_nu.plot_surface(x_nu, y_nu, z_nu, facecolors=plt.cm.get_cmap(colormap_mapping[selected_colormap_nu])(norm_nu), alpha=0.8)
    ax_nu.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.2)
    ax_nu.scatter([max_nu_point[0], min_nu_point[0]], [max_nu_point[1], min_nu_point[1]], [max_nu_point[2], min_nu_point[2]], color='red', s=50)
    ax_nu.text(max_nu_point[0], max_nu_point[1], max_nu_point[2], f'Max: {nu_values[max_nu_idx]:.3f}', size=10)
    ax_nu.text(min_nu_point[0], min_nu_point[1], min_nu_point[2], f'Min: {nu_values[min_nu_idx]:.3f}', size=10)
    ax_nu.set_xlabel('X')
    ax_nu.set_ylabel('Y')
    ax_nu.set_zlabel('Z')
    ax_nu.set_title(f"Poisson's Ratio for {material_name} (Matplotlib)")
    ax_nu.set_box_aspect([1,1,1])
    ax_nu.set_xlim([-1.2, 1.2])
    ax_nu.set_ylim([-1.2, 1.2])
    ax_nu.set_zlim([-1.2, 1.2])
    st.pyplot(fig_mpl_nu)
    
    st.write(f"Poisson's Ratio (ν12 along [100]): {nu_12:.3f}")
    st.write(f"Maximum Poisson's Ratio: {nu_values[max_nu_idx]:.3f}")
    st.write(f"Minimum Poisson's Ratio: {nu_values[min_nu_idx]:.3f}")

# Shear Modulus Tab
with tab3:
    st.subheader(f"Shear Modulus for {material_name}")
    # Plotly
    fig_G = go.Figure()
    fig_G.add_trace(go.Surface(
        x=x_G, y=y_G, z=z_G, surfacecolor=G_values, colorscale=selected_colormap_G,
        colorbar=dict(title="Shear Modulus (GPa)"), showscale=True
    ))
    fig_G.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere, surfacecolor=np.ones_like(x_sphere),
        colorscale=[[0, 'rgba(128, 128, 128, 0.2)'], [1, 'rgba(128, 128, 128, 0.2)']], showscale=False
    ))
    max_G_idx = np.unravel_index(np.argmax(G_values), G_values.shape)
    min_G_idx = np.unravel_index(np.argmin(G_values), G_values.shape)
    max_G_point = [x_G[max_G_idx], y_G[max_G_idx], z_G[max_G_idx]]
    min_G_point = [x_G[min_G_idx], y_G[min_G_idx], z_G[min_G_idx]]
    fig_G.add_trace(go.Scatter3d(
        x=[max_G_point[0], min_G_point[0]], y=[max_G_point[1], min_G_point[1]], z=[max_G_point[2], min_G_point[2]],
        mode='markers+text', text=[f'Max: {G_values[max_G_idx]:.2f} GPa', f'Min: {G_values[min_G_idx]:.2f} GPa'],
        textposition='top center', marker=dict(size=5, color='red')
    ))
    fig_G.update_layout(
        title=f"Shear Modulus (GPa) for {material_name} (Plotly)",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode='cube',
                   xaxis=dict(range=[-1.2, 1.2]), yaxis=dict(range=[-1.2, 1.2]), zaxis=dict(range=[-1.2, 1.2])),
        width=600, height=600
    )
    st.plotly_chart(fig_G)
    
    # Matplotlib
    fig_mpl_G = plt.figure(figsize=(6, 6))
    ax_G = fig_mpl_G.add_subplot(111, projection='3d')
    ax_G.plot_surface(x_G, y_G, z_G, facecolors=plt.cm.get_cmap(colormap_mapping[selected_colormap_G])(G_values/G_max), alpha=0.8)
    ax_G.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.2)
    ax_G.scatter([max_G_point[0], min_G_point[0]], [max_G_point[1], min_G_point[1]], [max_G_point[2], min_G_point[2]], color='red', s=50)
    ax_G.text(max_G_point[0], max_G_point[1], max_G_point[2], f'Max: {G_values[max_G_idx]:.2f} GPa', size=10)
    ax_G.text(min_G_point[0], min_G_point[1], min_G_point[2], f'Min: {G_values[min_G_idx]:.2f} GPa', size=10)
    ax_G.set_xlabel('X')
    ax_G.set_ylabel('Y')
    ax_G.set_zlabel('Z')
    ax_G.set_title(f"Shear Modulus (GPa) for {material_name} (Matplotlib)")
    ax_G.set_box_aspect([1,1,1])
    ax_G.set_xlim([-1.2, 1.2])
    ax_G.set_ylim([-1.2, 1.2])
    ax_G.set_zlim([-1.2, 1.2])
    st.pyplot(fig_mpl_G)
    
    st.write(f"Shear Modulus (G44 along [100]): {C[3,3]:.2f} GPa")
    st.write(f"Maximum Shear Modulus: {G_values[max_G_idx]:.2f} GPa")
    st.write(f"Minimum Shear Modulus: {G_values[min_G_idx]:.2f} GPa")

# Linear Compressibility Tab
with tab4:
    st.subheader(f"Linear Compressibility for {material_name}")
    # Plotly
    fig_beta = go.Figure()
    fig_beta.add_trace(go.Surface(
        x=x_beta, y=y_beta, z=z_beta, surfacecolor=beta_values, colorscale=selected_colormap_beta,
        colorbar=dict(title="Linear Compressibility (GPa⁻¹)"), showscale=True
    ))
    fig_beta.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere, surfacecolor=np.ones_like(x_sphere),
        colorscale=[[0, 'rgba(128, 128, 128, 0.2)'], [1, 'rgba(128, 128, 128, 0.2)']], showscale=False
    ))
    max_beta_idx = np.unravel_index(np.argmax(beta_values), beta_values.shape)
    min_beta_idx = np.unravel_index(np.argmin(beta_values), beta_values.shape)
    max_beta_point = [x_beta[max_beta_idx], y_beta[max_beta_idx], z_beta[max_beta_idx]]
    min_beta_point = [x_beta[min_beta_idx], y_beta[min_beta_idx], z_beta[min_beta_idx]]
    fig_beta.add_trace(go.Scatter3d(
        x=[max_beta_point[0], min_beta_point[0]], y=[max_beta_point[1], min_beta_point[1]], z=[max_beta_point[2], min_beta_point[2]],
        mode='markers+text', text=[f'Max: {beta_values[max_beta_idx]:.3f} GPa⁻¹', f'Min: {beta_values[min_beta_idx]:.3f} GPa⁻¹'],
        textposition='top center', marker=dict(size=5, color='red')
    ))
    fig_beta.update_layout(
        title=f"Linear Compressibility (GPa⁻¹) for {material_name} (Plotly)",
        scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode='cube',
                   xaxis=dict(range=[-1.2, 1.2]), yaxis=dict(range=[-1.2, 1.2]), zaxis=dict(range=[-1.2, 1.2])),
        width=600, height=600
    )
    st.plotly_chart(fig_beta)
    
    # Matplotlib
    fig_mpl_beta = plt.figure(figsize=(6, 6))
    ax_beta = fig_mpl_beta.add_subplot(111, projection='3d')
    norm_beta = (beta_values - np.min(beta_values)) / (np.max(beta_values) - np.min(beta_values)) if np.max(beta_values) != np.min(beta_values) else beta_values
    ax_beta.plot_surface(x_beta, y_beta, z_beta, facecolors=plt.cm.get_cmap(colormap_mapping[selected_colormap_beta])(norm_beta), alpha=0.8)
    ax_beta.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.2)
    ax_beta.scatter([max_beta_point[0], min_beta_point[0]], [max_beta_point[1], min_beta_point[1]], [max_beta_point[2], min_beta_point[2]], color='red', s=50)
    ax_beta.text(max_beta_point[0], max_beta_point[1], max_beta_point[2], f'Max: {beta_values[max_beta_idx]:.3f} GPa⁻¹', size=10)
    ax_beta.text(min_beta_point[0], min_beta_point[1], min_beta_point[2], f'Min: {beta_values[min_beta_idx]:.3f} GPa⁻¹', size=10)
    ax_beta.set_xlabel('X')
    ax_beta.set_ylabel('Y')
    ax_beta.set_zlabel('Z')
    ax_beta.set_title(f"Linear Compressibility (GPa⁻¹) for {material_name} (Matplotlib)")
    ax_beta.set_box_aspect([1,1,1])
    ax_beta.set_xlim([-1.2, 1.2])
    ax_beta.set_ylim([-1.2, 1.2])
    ax_beta.set_zlim([-1.2, 1.2])
    st.pyplot(fig_mpl_beta)
    
    st.write(f"Linear Compressibility (along [100]): {(S[0,0] + S[0,1] + S[0,2]):.3f} GPa⁻¹")
    st.write(f"Maximum Linear Compressibility: {beta_values[max_beta_idx]:.3f} GPa⁻¹")
    st.write(f"Minimum Linear Compressibility: {beta_values[min_beta_idx]:.3f} GPa⁻¹")
    
# Instructions

# LaTeX description of stiffness matrix and formulas
st.subheader("Formulas of Young's Modulus, Poisson's Ratio, Shear Modulus and Linear Compressibility")

st.markdown(r"""
# Elastic Properties:

**Young's Modulus (E):** Resistance to uniaxial deformation along direction $(l, m, n)$.

$$
E = \frac{1}{\mathbf{a}^T \mathbf{S} \mathbf{a}}, \quad
\mathbf{a} = \begin{bmatrix} l^2 & m^2 & n^2 & 2 m n & 2 n l & 2 l m \end{bmatrix}^T, \quad
\mathbf{S} = \mathbf{C}^{-1}
$$

Along [100]:

$$
E = \frac{1}{S_{11}}.
$$

---

**Poisson's Ratio ($\nu$):** Ratio of transverse to axial strain for direction $(l, m, n)$ and perpendicular direction $(l', m', n')$.

$$
\nu = -\frac{\mathbf{a}^T \mathbf{S} \mathbf{b}}{\mathbf{a}^T \mathbf{S} \mathbf{a}}, \quad
\mathbf{b} = \begin{bmatrix} l'^2 & m'^2 & n'^2 & 2 m' n' & 2 n' l' & 2 l' m' \end{bmatrix}^T, \quad
l l' + m m' + n n' = 0
$$

Along [100]:

$$
\nu_{12} = -\frac{S_{12}}{S_{11}}.
$$

---

**Shear Modulus (G):** Resistance to shear deformation in a plane with normal $(l', m', n')$ and shear direction $(l, m, n)$.

$$
G = \frac{1}{4 (\mathbf{b}^T \mathbf{S} \mathbf{b})}, \quad
\mathbf{b} = \begin{bmatrix}
l l' & m m' & n n' & m n' + m' n & n l' + n' l & l m' + l' m
\end{bmatrix}^T
$$

---

**Linear Compressibility ($\beta$):** Relative length change under hydrostatic pressure along $(l, m, n)$.

$$
\beta = \mathbf{a}^T \mathbf{S} \mathbf{c}, \quad
\mathbf{a} = \begin{bmatrix} l^2 & m^2 & n^2 & 0 & 0 & 0 \end{bmatrix}^T, \quad
\mathbf{c} = \begin{bmatrix} 1 & 1 & 1 & 0 & 0 & 0 \end{bmatrix}^T
$$
""", unsafe_allow_html=True)

