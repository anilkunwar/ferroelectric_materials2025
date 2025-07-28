import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.linalg import inv

# Initialize session state for input synchronization
if 'piezo_input' not in st.session_state:
    st.session_state.piezo_input = """0 0 0 0 0.2926 0
0 0 0 0.2926 0 0
0 0 0 0.6772 0.6772 3.448"""
if 'stiffness_input' not in st.session_state:
    st.session_state.stiffness_input = """260 108 77 0 0 0
108 260 77 0 0 0
77 77 89 0 0 0
0 0 0 31 0 0
0 0 0 0 31 0
0 0 0 0 0 116"""
if 'material_name' not in st.session_state:
    st.session_state.material_name = "BaTiO3 (Tetragonal)"

# Streamlit app title
st.title(f"3D Visualization of Piezoelectric Coefficient for {st.session_state.material_name}")

# Material name input
st.sidebar.subheader("Material Name")
material_name = st.sidebar.text_input("Enter material name", value=st.session_state.material_name, key="material_name")

# LaTeX description of piezoelectric tensor
st.subheader("Piezoelectric Tensor")
st.latex(r"""
\mathbf{e} = \begin{pmatrix}
e_{11} & e_{12} & e_{13} & e_{14} & e_{15} & e_{16} \\
e_{21} & e_{22} & e_{23} & e_{24} & e_{25} & e_{26} \\
e_{31} & e_{32} & e_{33} & e_{34} & e_{35} & e_{36}
\end{pmatrix}
""")

# Input for piezoelectric and stiffness matrices
st.sidebar.subheader(f"Input Matrices for {material_name}")
st.sidebar.markdown("""
Enter the 3x6 piezoelectric tensor (in C/m²) and 6x6 stiffness matrix (in GPa) below. Paste the full matrices in the textareas (rows separated by newlines, values by spaces).
""")

piezo_input = st.sidebar.text_area("Piezoelectric Tensor (3x6, C/m²)", 
                                  value=st.session_state.piezo_input, 
                                  key="piezo_input")

stiffness_input = st.sidebar.text_area("Stiffness Matrix (6x6, GPa)", 
                                      value=st.session_state.stiffness_input, 
                                      key="stiffness_input")

# Parse input matrices
try:
    e = np.array([[float(x) for x in row.split()] for row in piezo_input.strip().split('\n')])
    if e.shape != (3, 6):
        st.error("Piezoelectric tensor must be 3x6. Please check the input.")
        st.stop()
except ValueError:
    st.error("Invalid piezoelectric tensor input. Please enter numerical values separated by spaces.")
    st.stop()

try:
    C = np.array([[float(x) for x in row.split()] for row in stiffness_input.strip().split('\n')])
    if C.shape != (6, 6):
        st.error("Stiffness matrix must be 6x6. Please check the input.")
        st.stop()
except ValueError:
    st.error("Invalid stiffness matrix input. Please enter numerical values separated by spaces.")
    st.stop()

# Compute compliance matrix
try:
    S = inv(C)
except np.linalg.LinAlgError:
    st.error("Stiffness matrix is singular or invalid. Please check the input.")
    st.stop()

# Compute piezoelectric coefficient tensor d = e * S
d = e @ S  # Units: C/N

# Colormap selection
colormaps = [
    'Viridis', 'Cividis', 'Plasma', 'Inferno', 'Magma', 'Hot', 'Cool', 'Rainbow', 'Jet',
    'Blues', 'Greens', 'Reds', 'Purples', 'Oranges', 'Greys', 'YlOrRd', 'YlOrBr', 'YlGn',
    'RdBu', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
    'Coolwarm', 'Bwr', 'Seismic'
]
colormap_mapping = {
    'Viridis': 'viridis', 'Cividis': 'cividis', 'Plasma': 'plasma', 'Inferno': 'inferno',
    'Magma': 'magma', 'Hot': 'hot', 'Cool': 'cool', 'Rainbow': 'rainbow', 'Jet': 'jet',
    'Blues': 'Blues', 'Greens': 'Greens', 'Reds': 'Reds', 'Purples': 'Purples',
    'Oranges': 'Oranges', 'Greys': 'Greys', 'YlOrRd': 'YlOrRd', 'YlOrBr': 'YlOrBr',
    'YlGn': 'YlGn', 'RdBu': 'RdBu', 'PiYG': 'PiYG', 'PRGn': 'PRGn', 'BrBG': 'BrBG',
    'PuOr': 'PuOr', 'RdGy': 'RdGy', 'RdYlBu': 'RdYlBu', 'RdYlGn': 'RdYlGn',
    'Spectral': 'Spectral', 'Coolwarm': 'coolwarm', 'Bwr': 'bwr', 'Seismic': 'seismic'
}
st.sidebar.subheader("Select Colormap")
selected_colormap_d33 = st.sidebar.selectbox("Colormap for Piezoelectric Coefficient", colormaps, index=0, key="colormap_d33")

# Function to compute effective piezoelectric coefficient
def compute_d33_eff(direction, d):
    l, m, n = direction / np.linalg.norm(direction)
    n_vec = np.array([l, m, n])
    a = np.array([l**2, m**2, n**2, 2*m*n, 2*n*l, 2*l*m])
    d33_eff = n_vec.T @ d @ a  # Units: C/N
    return d33_eff * 1e12  # Convert to pC/N for typical units

# Generate points for 3D plot
theta = np.linspace(0, np.pi, 50)
phi = np.linspace(0, 2 * np.pi, 50)
theta, phi = np.meshgrid(theta, phi)
x = np.sin(theta) * np.cos(phi)
y = np.sin(theta) * np.sin(phi)
z = np.cos(theta)

# Compute d33_eff
d33_values = np.zeros_like(x)
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        direction = np.array([x[i, j], y[i, j], z[i, j]])
        d33_values[i, j] = compute_d33_eff(direction, d)

# Validate piezoelectric coefficient
d33_001 = d[2, 2] * 1e12  # Along [001], in pC/N
if abs(d33_001) > 1000:
    st.warning(f"Piezoelectric coefficient d33 along [001] is {d33_001:.2f} pC/N, which is unusually large. Typical values are below 1000 pC/N.")
if np.max(np.abs(d33_values)) > 1000 or np.min(d33_values) < -1000:
    st.warning(f"Directional d33 range [{np.min(d33_values):.2f}, {np.max(d33_values):.2f}] pC/N is unusual. Typical values are within ±1000 pC/N.")

# Scale coordinates for visualization
d33_max = np.max(np.abs(d33_values))
x_d33 = x * np.abs(d33_values) / d33_max
y_d33 = y * np.abs(d33_values) / d33_max
z_d33 = z * np.abs(d33_values) / d33_max
x_sphere = x
y_sphere = y
z_sphere = z

# Plotting
st.subheader(f"Effective Piezoelectric Coefficient (d33) for {material_name}")

# Plotly
fig_d33 = go.Figure()
fig_d33.add_trace(go.Surface(
    x=x_d33, y=y_d33, z=z_d33, surfacecolor=d33_values, colorscale=selected_colormap_d33,
    colorbar=dict(title="d33 (pC/N)"), showscale=True
))
fig_d33.add_trace(go.Surface(
    x=x_sphere, y=y_sphere, z=z_sphere, surfacecolor=np.ones_like(x_sphere),
    colorscale=[[0, 'rgba(128, 128, 128, 0.2)'], [1, 'rgba(128, 128, 128, 0.2)']], showscale=False
))
max_d33_idx = np.unravel_index(np.argmax(d33_values), d33_values.shape)
min_d33_idx = np.unravel_index(np.argmin(d33_values), d33_values.shape)
max_d33_point = [x_d33[max_d33_idx], y_d33[max_d33_idx], z_d33[max_d33_idx]]
min_d33_point = [x_d33[min_d33_idx], y_d33[min_d33_idx], z_d33[min_d33_idx]]
fig_d33.add_trace(go.Scatter3d(
    x=[max_d33_point[0], min_d33_point[0]], y=[max_d33_point[1], min_d33_point[1]], z=[max_d33_point[2], min_d33_point[2]],
    mode='markers+text', text=[f'Max: {d33_values[max_d33_idx]:.2f} pC/N', f'Min: {d33_values[min_d33_idx]:.2f} pC/N'],
    textposition='top center', marker=dict(size=5, color='red')
))
fig_d33.update_layout(
    title=f"Effective Piezoelectric Coefficient (pC/N) for {material_name} (Plotly)",
    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode='cube',
               xaxis=dict(range=[-1.2, 1.2]), yaxis=dict(range=[-1.2, 1.2]), zaxis=dict(range=[-1.2, 1.2])),
    width=600, height=600
)
st.plotly_chart(fig_d33)

# Matplotlib
fig_mpl_d33 = plt.figure(figsize=(6, 6))
ax_d33 = fig_mpl_d33.add_subplot(111, projection='3d')
norm_d33 = (d33_values - np.min(d33_values)) / (np.max(d33_values) - np.min(d33_values)) if np.max(d33_values) != np.min(d33_values) else d33_values
ax_d33.plot_surface(x_d33, y_d33, z_d33, facecolors=plt.cm.get_cmap(colormap_mapping[selected_colormap_d33])(norm_d33), alpha=0.8)
ax_d33.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.2)
ax_d33.scatter([max_d33_point[0], min_d33_point[0]], [max_d33_point[1], min_d33_point[1]], [max_d33_point[2], min_d33_point[2]], color='red', s=50)
ax_d33.text(max_d33_point[0], max_d33_point[1], max_d33_point[2], f'Max: {d33_values[max_d33_idx]:.2f} pC/N', size=10)
ax_d33.text(min_d33_point[0], min_d33_point[1], min_d33_point[2], f'Min: {d33_values[min_d33_idx]:.2f} pC/N', size=10)
ax_d33.set_xlabel('X')
ax_d33.set_ylabel('Y')
ax_d33.set_zlabel('Z')
ax_d33.set_title(f"Effective Piezoelectric Coefficient (pC/N) for {material_name} (Matplotlib)")
ax_d33.set_box_aspect([1, 1, 1])
ax_d33.set_xlim([-1.2, 1.2])
ax_d33.set_ylim([-1.2, 1.2])
ax_d33.set_zlim([-1.2, 1.2])
st.pyplot(fig_mpl_d33)

st.write(f"Piezoelectric Coefficient d33 (along [001]): {d33_001:.2f} pC/N")
st.write(f"Maximum d33: {d33_values[max_d33_idx]:.2f} pC/N")
st.write(f"Minimum d33: {d33_values[min_d33_idx]:.2f} pC/N")

# LaTeX formulas
st.subheader("Formula for Effective Piezoelectric Coefficient")
st.markdown(r"""
The effective longitudinal piezoelectric coefficient \( d_{33}^{\text{eff}} \) along direction \( \mathbf{n} = [l, m, n] \) is given by:

\[
d_{33}^{\text{eff}} = \mathbf{n}^T \mathbf{d} \mathbf{a}, \quad \mathbf{n} = \begin{bmatrix} l \\ m \\ n \end{bmatrix}, \quad \mathbf{a} = \begin{bmatrix} l^2 \\ m^2 \\ n^2 \\ 2mn \\ 2nl \\ 2lm \end{bmatrix}, \quad \mathbf{d} = \mathbf{e} \mathbf{S}
\]

where:

- \( \mathbf{e} \) is the 3×6 piezoelectric tensor (C/m²).
- \( \mathbf{S} = \mathbf{C}^{-1} \) is the 6×6 compliance matrix (GPa⁻¹).
- \( \mathbf{d} \) is the piezoelectric coefficient tensor (C/N).
- Along [001]: \( d_{33} = d_{333} \).

The result is converted to pC/N for visualization (1 C/N = \(10^{12}\) pC/N).
""", unsafe_allow_html=True)


