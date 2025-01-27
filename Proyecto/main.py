import streamlit as st
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.stats.stattools import durbin_watson
from scipy import stats
from statsmodels.compat import lzip
import statsmodels.stats.api as sms
from scipy.stats import normaltest
from statsmodels.formula.api import ols
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config( 
    page_title="Impacto del trabajo remoto en la salud mental",
    page_icon="./public/logo.png"
)

# Título del proyecto
st.sidebar.title('Proyecto Final de Estadística')
st.sidebar.subheader('Impacto del trabajo remoto en la salud mental')
st.sidebar.image('./public/logo.png', use_container_width=True)

# Integrantes
st.sidebar.markdown('''
**Integrantes:**

- Claudia Hernández Pérez         C-312
- Joel Aparicio Tamayo            C-312
- Kendry Javier del Pino Barbosa  C-312
''')

st.image('./public/image.jpg', use_container_width=True)

# Descripción del proyecto
st.markdown('''
A medida que el trabajo remoto se convierte en la nueva norma, es esencial comprender su impacto en el bienestar mental de los empleados. Este conjunto de datos analiza cómo trabajar de forma remota afecta los niveles de estrés, el equilibrio entre la vida laboral y personal, y las condiciones de salud mental en diversas industrias y regiones.

Con 5,000 registros recopilados de empleados de todo el mundo, este conjunto de datos proporciona información valiosa sobre áreas clave como la ubicación del trabajo (remoto, híbrido, en sitio), los niveles de estrés, el acceso a recursos de salud mental y la satisfacción laboral. Está diseñado para ayudar a investigadores, profesionales de recursos humanos y empresas a evaluar la creciente influencia del trabajo remoto en la productividad y el bienestar.
''')

# Cargar los datos
impact = pd.read_csv('./data/Impact_of_Remote_Work_on_Mental_Health.csv')
st.dataframe(impact)

st.markdown('---')

st.markdown("<h1 style='text-align: center;'>Análisis Exploratorio de Datos</h1>", unsafe_allow_html=True)
st.markdown('''El estudio se ha realizado en una muestra de 5000 trabajadores con niveles de experiencia bastante equilibrados, 
excepto algunos picos cerca del primer año, 10, 20, 27 y 35 años. Y de todas partes del mundo''')

col1, col2 = st.columns(2)

with col1:
    # Histograma de Años de Experiencia
    st.markdown('#### Distribución de Años de Experiencia')
    fig, ax = plt.subplots()
    ax.hist(impact['Years_of_Experience'].dropna(), bins=30, edgecolor='black')
    ax.set_xlabel('Años de experiencia')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Distribución de Años de Experiencia')
    st.pyplot(fig)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gráfico de pastel por nivel de estrés
    st.markdown('#### Distribución por Nivel de Estrés')    
    stress_counts = impact.groupby('Stress_Level')['Stress_Level'].agg('count')
    labels = ['Alto', 'Bajo', 'Medio']
    sizes = stress_counts.values
    colors = ['#e74c3c', '#2ecc71', '#f39c12']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Gráfico de pastel por satisfacción con el trabajo remoto
    st.markdown('#### Distribución por Satisfacción con el Trabajo Remoto')
    satisfaction_counts = impact.groupby('Satisfaction_with_Remote_Work')['Satisfaction_with_Remote_Work'].agg('count')
    # Datos para el gráfico de pastel
    labels = ['Neutral', 'Satisfechos', 'Insatisfechos']
    sizes = satisfaction_counts.values
    colors = ['gold', 'yellowgreen', 'lightcoral']
    explode = (0, 0, 0.05)
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

with col2:
    # Gráfico de pastel por región
    st.markdown('#### Distribución por Región')
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    region_counts = impact.groupby('Region')['Region'].agg('count')
    labels = ['África', 'Asia', 'Europa', 'América del Norte', 'Oceanía', 'América del Sur']
    sizes = region_counts.values
    colors = ['#2E8B57', '#FF6347', '#4682B4', '#FFA500', '#87CEFA', '#FFD700']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Gráfico de pastel por ubicación laboral
    st.markdown("<h4 style='text-align: center;'>Distribución por Ubicación Laboral</h4>", unsafe_allow_html=True)
    location_counts = impact.groupby('Work_Location')['Work_Location'].agg('count')
    labels = ['Híbrido', 'En el trabajo', 'En casa']
    sizes = location_counts.values
    colors = ['#b4a7d6', '#6fa8dc', '#a8d08d']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)
    
    st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
    
    st.markdown('''
    ### ¿Pero estos niveles de satisfacción que impacto puede tener en la salud mental de la población a nivel mundial?
    Apoyándonos en que la minoría de nuestros datos está descontenta, ¿se podrá decir que la minoría de la población ha tenido que acceder a recursos de salud mental?

    Probemos entonces que menos del 50% de los trabajadores a nivel mundial ha tenido que acceder a recursos de salud mental como resultado del trabajo remoto con un nivel de confianza de 95%.            
    ''')

    
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Prueba de Hipótesis sobre Acceso a Recursos de Salud Mental</h2>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
    
col1, col2 = st.columns(2)

with col1: 
    st.markdown(r'''
    $$H_0: \rho \geq 0.5$$

    $$H_1: \rho < 0.5$$

    Estadígrafo: $$\rho = \frac{\overline{\rho} - \rho_0}{\sqrt{\rho_0(1 - \rho_0)}} \sqrt{n}$$

    Región crítica (n > 30): $$\rho < -Z_{1-\alpha}$$
    ''')
    
    # Prueba de hipótesis sobre acceso a recursos de salud mental
    resources_counts = impact.groupby('Access_to_Mental_Health_Resources')['Access_to_Mental_Health_Resources'].agg('count')
    access_workers = resources_counts['Yes']
    total_workers = len(impact)
    p_hat = access_workers / total_workers
    p_0 = 0.5
    alpha = 0.05
    z = (p_hat - p_0) / np.sqrt(p_0 * (1 - p_0) / total_workers)
    z_alpha = stats.norm.ppf(1 - alpha)
    if z < -z_alpha:
        st.write("Rechazamos la hipótesis nula H0")
    else:
        st.write("No podemos rechazar la hipótesis nula H0")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown('Como se muestra, los niveles de satisfacción están directamente relacionados con la afectación de la salud mental de las personas en el trabajo remoto. Por lo que podría decirse que aunque no ha resultado beneficiosa para la mayoría, los criterios están muy cercanos.')

with col2:    
    # Gráfico de pastel por acceso a recursos de salud mental
    # Datos para el gráfico de pastel
    labels = ['Sí', 'No']
    sizes = resources_counts.values
    colors = ['yellowgreen', 'lightcoral']
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')
    st.pyplot(fig)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Análisis de la Edad de los Encuestados</h2>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
    
col1, col2 = st.columns(2)

with col1:
    # Boxplot de la Edad de los Encuestados
    st.markdown('#### Boxplot de la Edad de los Encuestados')
    fig, ax = plt.subplots()
    ax.boxplot(impact['Age'].dropna())
    ax.set_ylabel('Edad')
    ax.set_title('Boxplot de la Edad de los Encuestados')
    st.pyplot(fig)
    
with col2:
    # Histograma de la Edad de los Encuestados
    st.markdown('#### Histograma de la Edad de los Encuestados')
    fig, ax = plt.subplots()
    ax.hist(impact['Age'].dropna(), bins=30, edgecolor='black')
    ax.set_xlabel('Edad')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Histograma de la Edad de los Encuestados')
    st.pyplot(fig)
    
st.markdown('''
El histograma parece mostrar que las edades distribuyen normal. Tomemos dicha hipótesis e intentemos probarla: 

El siguiente bloque de código se enfoca en probar la hipótesis de normalidad de la distribución de las edades de los encuestados, utilizando la prueba de normalidad de D'Agostino-Pearson y visualizando los resultados a través de un histograma y un gráfico Q-Q.

Un gráfico Q-Q (Quantile-Quantile plot) es una herramienta gráfica utilizada para comparar la distribución de un conjunto de datos con una distribución teórica, como la distribución normal. El objetivo principal de un gráfico Q-Q es evaluar si los datos siguen una distribución específica.

Línea recta: Si los puntos en el gráfico Q-Q se alinean aproximadamente a lo largo de una línea recta, esto sugiere que los datos siguen la distribución teórica.

Desviaciones: Desviaciones significativas de la línea recta pueden indicar que los datos no siguen la distribución teórica.            
''')

# Prueba de normalidad de la Edad
st.markdown("<h3 style='text-align: center;'>Prueba de Normalidad de la Edad</h3>", unsafe_allow_html=True)
k2, p = stats.normaltest(impact['Age'])
st.write(f'Estadístico K2: {k2}')
st.write(f'Valor p: {p}')
if p > alpha:
    st.write("No se puede rechazar la hipótesis nula: los datos parecen provenir de una distribución normal.")
else:
    st.write("Se rechaza la hipótesis nula: los datos no parecen provenir de una distribución normal.")

st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Crear el histograma
    fig, ax = plt.subplots()
    ax.hist(impact['Age'], bins=30, edgecolor='black', density=True)
    ax.set_xlabel('Valor')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Histograma de datos generados')

    # Superponer la curva de distribución normal
    xmin, xmax = ax.get_xlim()
    mean, std_dev = np.mean(impact['Age']), np.std(impact['Age'])
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean, std_dev)
    ax.plot(x, p, 'k', linewidth=2)

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)

with col2:
    # Gráfico Q-Q de la Edad
    fig, ax = plt.subplots()
    stats.probplot(impact['Age'], dist="norm", plot=ax)
    ax.set_title('Gráfico Q-Q')
    st.pyplot(fig)
    
st.markdown('Esto se traduce a que los encuestados como son trabajadores, lo más \'normal\' que podía pasar es que las edades más populares estuviesen en los datos centrales y los menos en las esquinas, dado que sobre los 20 se empieza en el mundo laboral y ya luego cerca de los 50 es menos frecuente que continúe su vida profesional.')

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Estimadores Puntuales</h3>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(r'Utilizaremos como estimador puntual de la media a $\overline{x} = \frac{1}{n}\sum{x_i}$, pues es insesgado, consistente y eficiente.')
    mean_age = np.mean(impact['Age'])
    st.write(f"Estimador puntual de la media: {mean_age}")

with col2:
    st.markdown(r'Utilizaremos como estimador puntual de la varianza a $S^2 = \frac{1}{n-1}\sum{(x_i - \overline{x})^2}$, pues es insesgado, consistente y eficiente.')
    std_dev_age = np.std(impact['Age'], ddof=1)
    st.write(f"Estimador puntual de la varianza: {std_dev_age**2}")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Intervalos de Confianza</h3>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1: 
    st.markdown('Con los estimadores puntuales de la media y la varianza calculados, podemos también estimar la media poblacional por intervalo de confianza. Para ello utilicemos una seguridad del 95%. Como sabemos que la edad distribuye normal podemos utilizar el intervalo de confianza siguiente:')
    st.markdown(r'$$\mu \in [\overline{x} - \frac{S}{\sqrt{n}}Z_{1-\frac{\alpha}{2}},\overline{x} + \frac{S}{\sqrt{n}}Z_{1-\frac{\alpha}{2}}]$$')
    n = len(impact['Age'])
    Z = stats.norm.ppf(1 - alpha/2)
    e = (std_dev_age / np.sqrt(n)) * Z
    lower_bound = mean_age - e
    upper_bound = mean_age + e
    st.write(f"El intervalo de confianza de la media es [{lower_bound}, {upper_bound}] con 95% de seguridad.")

with col2:
    st.markdown('Con los estimadores puntuales de la media y la varianza calculados, podemos también estimar la varianza poblacional por intervalo de confianza. Para ello utilicemos una seguridad del 95%. Como sabemos que la edad distribuye normal podemos utilizar el intervalo de confianza siguiente:')
    st.markdown(r'$$\sigma^2 \in [\frac{(n-1)S^2}{\chi^2_{1-\frac{\alpha}{2}}(n-1)},\frac{(n-1)S^2}{\chi^2_{\frac{\alpha}{2}}(n-1)}]$$')
    S2 = std_dev_age**2
    chi2_lower = stats.chi2.ppf(1 - alpha/2, n - 1)
    chi2_upper = stats.chi2.ppf(alpha/2, n - 1)
    lower_bound = (n - 1) * S2 / chi2_lower
    upper_bound = (n - 1) * S2 / chi2_upper
    st.write(f"El intervalo de confianza para la varianza poblacional es: [{lower_bound}, {upper_bound}]")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Análisis de las horas trabajadas por semana por género</h2>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Extraemos las horas trabajadas por semana y el género
data = impact[['Gender', 'Hours_Worked_Per_Week']]

# Creamos una tabla de contingencia
contingency_table = pd.crosstab(data['Gender'], data['Hours_Worked_Per_Week'])
st.write(contingency_table)

# Aumentamos el tamaño de la figura para darle más espacio al gráfico
fig, ax = plt.subplots(figsize=(14, 12))

# Creamos el heatmap con los géneros en el eje x y las horas en el eje y
sns.heatmap(contingency_table.T, annot=True, fmt="d", cmap="YlGnBu", ax=ax, 
            annot_kws={"size": 8}, cbar_kws={'shrink': 0.5})

# Configuramos el título y los ejes con tamaños de fuente adecuados
ax.set_xlabel('Género', fontsize=14)
ax.set_ylabel('Horas Trabajadas por Semana', fontsize=14)

# Rotamos las etiquetas del eje x para evitar solapamientos
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=12)

# Ajustamos los márgenes para que todo quepa correctamente
plt.tight_layout()

st.pyplot(fig)

st.markdown("<h3 style='text-align: center;'>Análisis de las horas trabajadas por semana de hombres vs no hombres</h3>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

men_hours = impact[impact['Gender'] == 'Male']['Hours_Worked_Per_Week']
k2_men, p_men = stats.normaltest(men_hours)
non_men_hours = impact[impact['Gender'] != 'Male']['Hours_Worked_Per_Week']
k2_non_men, p_non_men = stats.normaltest(non_men_hours)
    
with col1:
    container = st.container(border=True, key="men")
    container.write(f'Estadístico K2 (Hombres): {k2_men}')
    container.write(f'Valor p (Hombres): {p_men}')
    if p_men > alpha:
        container.write("No se puede rechazar la hipótesis nula para hombres: los datos parecen provenir de una distribución normal.")
    else:
        container.write("Se rechaza la hipótesis nula para hombres: los datos no parecen provenir de una distribución normal.")

    # Histograma y curva de distribución para hombres y no hombres
    fig, ax = plt.subplots()
    
    # Hombres
    ax.hist(men_hours, bins=30, edgecolor='black', density=True, alpha=0.5, label='Hombres', color='blue')
    mean_men, std_dev_men = np.mean(men_hours), np.std(men_hours)
    x_men = np.linspace(xmin, xmax, 100)
    p_men = stats.norm.pdf(x_men, mean_men, std_dev_men)
    ax.plot(x_men, p_men, 'b', linewidth=2, label='Curva Normal - Hombres')
    
    # No Hombres
    ax.hist(non_men_hours, bins=30, edgecolor='black', density=True, alpha=0.5, label='No Hombres', color='red')
    mean_non_men, std_dev_non_men = np.mean(non_men_hours), np.std(non_men_hours)
    x_non_men = np.linspace(xmin, xmax, 100)
    p_non_men = stats.norm.pdf(x_non_men, mean_non_men, std_dev_non_men)
    ax.plot(x_non_men, p_non_men, 'r', linewidth=2, label='Curva Normal - No Hombres')
    
    ax.set_xlabel('Horas Trabajadas por Semana')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Histograma y Curvas Normales para Hombres y No Hombres')
    ax.legend()
    st.pyplot(fig)
    
with col2:
    non_men_hours = impact[impact['Gender'] != 'Male']['Hours_Worked_Per_Week']
    k2_non_men, p_non_men = stats.normaltest(non_men_hours)
    container = st.container(border=True, key="no men")
    container.write(f'Estadístico K2 (No Hombres): {k2_non_men}')
    container.write(f'Valor p (No Hombres): {p_non_men}')
    if p_non_men > alpha:
        container.write("No se puede rechazar la hipótesis nula para no hombres: los datos parecen provenir de una distribución normal.")
    else:
        container.write("Se rechaza la hipótesis nula para no hombres: los datos no parecen provenir de una distribución normal.")

    # Gráfico Q-Q para hombres y no hombres
    fig, ax = plt.subplots()
    
    # Hombres
    stats.probplot(men_hours, dist="norm", plot=ax)
    ax.get_lines()[0].set_color('blue')  # Cambiar color de los puntos
    ax.get_lines()[1].set_color('blue')  # Cambiar color de la línea
    
    # No Hombres
    stats.probplot(non_men_hours, dist="norm", plot=ax)
    ax.get_lines()[2].set_color('red')  # Cambiar color de los puntos
    ax.get_lines()[3].set_color('red')  # Cambiar color de la línea
    
    ax.set_title('Gráfico Q-Q para Hombres y No Hombres')
    ax.legend(['Hombres', 'Línea Hombres', 'No Hombres', 'Línea No Hombres'])
    st.pyplot(fig)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Prueba de Hipótesis para Dos Poblaciones</h3>", unsafe_allow_html=True)
st.markdown('Una vez demostrado que las horas trabajadas por semana de los hombres y demas géneros distribuyen normal, hagamos una prueba de hipótesis para dos poblaciones: queremos probar que las varianzas de los hombres y demás géneros son distintas con un nivel de significancia del 95%.')
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1: 
    st.markdown(r'''
    **Hipótesis planteadas:**  
    $$H_0: \sigma_{\text{hombres}}^2 = \sigma_{\text{no\ hombres}}^2$$  
    $$H_1: \sigma_{\text{hombres}}^2 \neq \sigma_{\text{no\ hombres}}^2$$  

    **Estadígrafo:**  
    $$
    W = \frac{(N - k)}{(k - 1)} \cdot \frac{\sum_{i=1}^{k} n_i (\bar{X}_i - \bar{X})^2}{\sum_{i=1}^{k} \sum_{j=1}^{n_i} (X_{ij} - \bar{X}_i)^2}
    $$ 

    **Región Crítica (RH0):**  
    $$W > F_\alpha$$
    ''')

    stat, p_value_levene = stats.levene(men_hours, non_men_hours)
    st.write(f"Estadístico de Levene: {stat}")
    st.write(f"Valor p (Levene): {p_value_levene}")
    if p_value_levene < alpha:
        st.write("Rechazamos la hipótesis nula. Las varianzas son significativamente diferentes (Levene).")
    else:
        st.write("No podemos rechazar la hipótesis nula. No hay evidencia suficiente para decir que las varianzas son diferentes (Levene).")
    st.markdown('Una vez demostrado que las horas trabajadas por semana de los hombres y demas géneros distribuyen normal y que las varianzas son diferentes, hagamos una prueba de hipótesis para dos poblaciones: queremos probar que la media de trabajo de los hombres es superior a los demas géneros con un nivel de significancia del 95%.')

with col2:
    st.markdown(r'''
    **Hipótesis planteadas:**
    $$H_0: \mu_{\text{hombres}} \leq \mu_{\text{no hombres}}$$  
    $$H_1: \mu_{\text{hombres}} > \mu_{\text{no hombres}}$$  

    **Estadígrafo:**  

    $$T = \frac{\overline{X}_{\text{hombres}} - \overline{X}_{\text{no hombres}}}{\sqrt{\frac{S_{\text{hombres}}^2}{n_{\text{hombres}}} + \frac{S_{\text{no hombres}}^2}{n_{\text{no hombres}}}}}$$  

    **Grados de libertad aproximados:**  
    $$
    v = \left( \frac{(\frac{S_{\text{hombres}}^2}{n_{\text{hombres}}} + \frac{S_{\text{no\ hombres}}^2}{n_{\text{no\ hombres}}})^2}{\left(\frac{S_{\text{hombres}}^2}{n_{\text{hombres}}}\right) \left(\frac{1}{n_{\text{hombres}} + 1}\right) + \left(\frac{S_{\text{no\ hombres}}^2}{n_{\text{no\ hombres}}}\right) \left(\frac{1}{n_{\text{no\ hombres}} + 1}\right)} \right) - 2
    $$

    **Región crítica:**  
    $$T > T_{1-\alpha}(v)$$
    ''')

    mean_men = np.mean(men_hours)
    std_dev_men = np.std(men_hours, ddof=1)
    n_men = len(men_hours)
    mean_non_men = np.mean(non_men_hours)
    std_dev_non_men = np.std(non_men_hours, ddof=1)
    n_non_men = len(non_men_hours)
    T = (mean_men - mean_non_men) / np.sqrt((std_dev_men**2 / n_men) + (std_dev_non_men**2 / n_non_men))
    numerator = (std_dev_men**2 / n_men + std_dev_non_men**2 / n_non_men) ** 2
    denominator = ((std_dev_men**2 / n_men) / (n_men + 1)) + ((std_dev_non_men**2 / n_non_men) / (n_non_men + 1))
    degrees_of_freedom = numerator / denominator - 2
    t_critical = stats.t.ppf(1 - alpha, df=degrees_of_freedom)
    if T > t_critical:
        st.markdown("Se rechaza la hipótesis nula: la media de las horas trabajadas por los hombres es mayor que la de los otros géneros.")
    else:
        st.markdown("No se puede rechazar la hipótesis nula: la media de las horas trabajadas por los hombres no es significativamente mayor que la de los otros géneros.")
    st.markdown(f'Estadístico T: {T}')
    st.markdown(f'Grados de libertad: {degrees_of_freedom}')
    st.markdown(f'Valor crítico (t-student): {t_critical}')

col1, col2 = st.columns(2)

with col1: 
# Pair plot
    st.markdown('#### Pair Plot')

    # Ejemplo de datos (asegúrate de que men_hours y non_men_hours estén definidos)
    min_length = min(len(men_hours.values), len(non_men_hours.values))
    data = {
        'Hombres': men_hours.values.tolist()[:min_length],
        'Resto de Géneros': non_men_hours.values.tolist()[:min_length],
        'Grupo': ['Hombres'] * 635 + ['Resto de Géneros'] * 635
    }
    df = pd.DataFrame(data)

    # Crear el pair plot
    fig = sns.pairplot(df, hue='Grupo')

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)
    
with col2: 
    st.markdown('<br><br><br>', unsafe_allow_html=True)
    st.markdown('''
    La prueba de hipótesis indica que los hombres tienden a trabajar más horas de media a la semana que otros 
    géneros en esta muestra, subrayando la necesidad de abordar los posibles riesgos que este comportamiento 
    puede tener en su salud mental. Es fundamental que tanto los empleadores como los empleados consideren el 
    equilibrio entre el trabajo y la vida personal como una prioridad, para prevenir el estrés crónico, el 
    agotamiento y otros problemas relacionados con la salud mental. A largo plazo, promover una cultura laboral 
    que valore el bienestar de los empleados puede ser esencial para mejorar la salud mental de los hombres y 
    de la fuerza laboral en general.        
    ''',)

st.markdown("<h2 style='text-align: center;'>Análisis de correlación</h2>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Seleccionar solo las columnas necesarias
    df_grouped = impact.groupby('Number_of_Virtual_Meetings').agg(
        num_hours=pd.NamedAgg(column='Hours_Worked_Per_Week', aggfunc='count'),
        total_hours=pd.NamedAgg(column='Hours_Worked_Per_Week', aggfunc='sum')
    ).reset_index()

    df_grouped['average_hours'] = df_grouped['total_hours'] / df_grouped['num_hours']

    # Calcular la correlación
    correlation = df_grouped['Number_of_Virtual_Meetings'].corr(df_grouped['average_hours'])

    # Mostrar la correlación en Streamlit
    st.write(f"La correlación es: {correlation}")
    # Realizar un ajuste lineal
    slope, intercept = np.polyfit(df_grouped['Number_of_Virtual_Meetings'], df_grouped['average_hours'], 1)

    # Mostrar la pendiente y la intersección en Streamlit
    st.write(f"La pendiente es: {slope}, y la intersección es: {intercept}")

    # Graficar los datos y la línea de ajuste
    plt.figure(figsize=(16, 8))
    plt.scatter(df_grouped['Number_of_Virtual_Meetings'], df_grouped['average_hours'], color='blue')
    plt.plot(df_grouped['Number_of_Virtual_Meetings'], slope * df_grouped['Number_of_Virtual_Meetings'] + intercept, color='red')
    plt.title('Relación entre Reuniones Virtuales y Horas Trabajadas por Semana')
    plt.xlabel('Número de Reuniones Virtuales')
    plt.ylabel('Horas Promedio Trabajadas por Semana')
    plt.grid(True)
    st.pyplot(plt)

with col2:
    # Seleccionar solo las columnas necesarias
    df_grouped = impact.groupby('Age').agg(
        num_support=pd.NamedAgg(column='Company_Support_for_Remote_Work', aggfunc='count'),
        total_support=pd.NamedAgg(column='Company_Support_for_Remote_Work', aggfunc='sum')
    ).reset_index()

    df_grouped['average_support'] = df_grouped['total_support'] / df_grouped['num_support']

    # Calcular la correlación
    correlation = df_grouped['Age'].corr(df_grouped['average_support'])

    # Mostrar la correlación en Streamlit
    st.write(f"La correlación es: {correlation}")

    # Realizar un ajuste lineal
    slope, intercept = np.polyfit(df_grouped['Age'], df_grouped['average_support'], 1)

    # Mostrar la pendiente y la intersección en Streamlit
    st.write(f"La pendiente es: {slope}, y la intersección es: {intercept}")

    # Graficar los datos y la línea de ajuste
    plt.figure(figsize=(16, 8))
    plt.scatter(df_grouped['Age'], df_grouped['average_support'], color='blue')
    plt.plot(df_grouped['Age'], slope * df_grouped['Age'] + intercept, color='red')
    plt.title('Relación entre Edad y Apoyo de las Compañías al Trabajo Remoto')
    plt.xlabel('Edad')
    plt.ylabel('Apoyo Promedio de las Compañías al Trabajo Remoto')
    plt.grid(True)
    st.pyplot(plt)

st.markdown("<h2 style='text-align: center;'>Matriz de correlación</h2>", unsafe_allow_html=True)

# Seleccionar las columnas numéricas
numeric_col = ['Age', 'Years_of_Experience', 'Hours_Worked_Per_Week', 'Number_of_Virtual_Meetings', 
               'Work_Life_Balance_Rating', 'Social_Isolation_Rating', 'Company_Support_for_Remote_Work']

# Crear el heatmap
plt.figure(figsize=(10, 5))
heatmap = sns.heatmap(impact[numeric_col].corr(), annot=True, cmap='coolwarm')

# Mostrar el heatmap en Streamlit
st.pyplot(heatmap.figure)

# Variables independientes
X = impact[['Age', 'Years_of_Experience', 'Number_of_Virtual_Meetings', 'Hours_Worked_Per_Week', 
            'Work_Life_Balance_Rating', 'Social_Isolation_Rating']]

# Variable dependiente
y = impact['Company_Support_for_Remote_Work']

# Añadir la constante
X = sm.add_constant(X)

# Ajustar el Modelo de Regresión Lineal
model = sm.OLS(y, X).fit()

st.markdown("<h2 style='text-align: center;'>Análisis de Regresión Lineal</h2>", unsafe_allow_html=True)

# Mostrar el resumen del modelo en Streamlit
st.write(model.summary())

# Seleccionar solo las columnas necesarias
df_grouped = impact.groupby('Age').agg(
    num_support=pd.NamedAgg(column='Company_Support_for_Remote_Work', aggfunc='count'),
    total_support=pd.NamedAgg(column='Company_Support_for_Remote_Work', aggfunc='sum')
).reset_index()

df_grouped['average_support'] = df_grouped['total_support'] / df_grouped['num_support']

# Calcular la correlación
correlation = df_grouped['Age'].corr(df_grouped['average_support'])

st.markdown("<h2 style='text-align: center;'>Análisis de Regresión Lineal: Edad y Apoyo de las Compañías</h2>", unsafe_allow_html=True)   

# Mostrar la correlación en Streamlit
st.write(f"La correlación es: {correlation}")

# Realizar un ajuste lineal
slope, intercept = np.polyfit(df_grouped['Age'], df_grouped['average_support'], 1)

# Mostrar la pendiente y la intersección en Streamlit
st.write(f"La pendiente es: {slope}, y la intersección es: {intercept}")

col1, col2 = st.columns(2)

with col1:
    # Ajustar el Modelo de Regresión Lineal
    X = sm.add_constant(df_grouped['Age'])
    model = sm.OLS(df_grouped['average_support'], X).fit()

    # Obtener los residuos del modelo
    residuals = model.resid

    # Mostrar el resumen del modelo en Streamlit
    st.write(model.summary())

with col2:
    # Gráfico QQ de los residuos
    fig = sm.qqplot(residuals, line='s')
    plt.title('Gráfico QQ de los Residuos')
    st.pyplot(fig)
    
st.write("""
    Se tienen los siguientes resultados:

    - **R-cuadrado**: El valor de R-cuadrado es 0.337, lo que significa que aproximadamente el 33.7% de la variabilidad en average_support puede ser explicada por las variables independientes en el modelo.
    - **F-estadístico**: El valor de F-estadístico es 14.73 y el valor p asociado es muy pequeño (0.000621), lo que indica que al menos una de las variables independientes es significativamente diferente de cero en el nivel de confianza del 95%.
    - **Coeficientes**: Los coeficientes representan el cambio en la variable dependiente por cada cambio de una unidad en la variable independiente, manteniendo constantes las demás variables. Por ejemplo, por cada aumento de una unidad en Age, average_support aumenta en promedio 0.0229 unidades.
    - **p-value**: Los p-values para cada coeficiente indican si la variable es significativa en el modelo. Si el valor p es menor que 0.05, la variable es significativa. En este caso, Age es significativa.
    - **Omnibus/Prob(Omnibus)**: Prueba la hipótesis de que los residuos están normalmente distribuidos. Un valor de Prob(Omnibus) cercano a 1 indica que los residuos están normalmente distribuidos. En este caso, el valor es 0.024, lo que indica que los residuos no están perfectamente distribuidos normalmente.
    - **Durbin-Watson**: Prueba la existencia de autocorrelación en los residuos. Un valor cercano a 2 indica que no hay autocorrelación. En este caso, el valor es 1.910, lo que indica que no hay autocorrelación significativa.
    - **Jarque-Bera (JB)/Prob(JB)**: Prueba la hipótesis de que los residuos están normalmente distribuidos. Un valor de Prob(JB) cercano a 1 indica que los residuos están normalmente distribuidos. En este caso, el valor es 0.0562, lo que indica que los residuos están razonablemente distribuidos normalmente.
    - **Cond. No.**: Indica la multicolinealidad en los datos. Un número mayor a 30 puede indicar una fuerte multicolinealidad. En este caso, el valor es 146, lo que sugiere que puede haber alguna multicolinealidad en los datos.
""")


st.markdown("<h2 style='text-align: center;'>Análisis de supuestos</h2>", unsafe_allow_html=True)

# Obtener los residuos del modelo
residuals = model.resid

col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 style='text-align: center;'>1. Los errores son independientes</h3>", unsafe_allow_html=True)
    st.write("Test de Durbin Watson")

    st.markdown("$H_0:$ No existe correlación entre los residuos")
    st.markdown("$H_1:$ Los residuos están autocorrelacionados")

    # Test de Durbin-Watson para independencia de los errores
    dw_test = durbin_watson(residuals)

    st.write(f'Test de Durbin-Watson: {dw_test}')

    alpha = 0.5

    # Interpretación del test
    if 2 - alpha <= dw_test <= 2 + alpha:
        st.write("Los residuos no están correlacionados. Se cumple el supuesto de independencia.")
    elif dw_test > 2 + alpha:
        st.write("Hay una autocorrelación negativa en los residuos.")
    else:
        st.write("Hay una autocorrelación positiva en los residuos.")
                
with col2: 
    st.markdown("<h3 style='text-align: center;'>2. El valor esperado de los errores es cero</h3>", unsafe_allow_html=True)
    st.write("Test para la media de una población")

    st.markdown("$H_0: \mu_0 = 0$")
    st.markdown("$H_1: \mu_0 \\neq 0$")

    # Test para la media de una población
    t_stat, p_value = stats.ttest_1samp(residuals, 0)

    st.write(f"T-statistic: {t_stat:.5f}, P-value: {p_value:.5f}")

    if p_value < 0.05:
        st.write("Hay evidencia para rechazar la hipótesis nula de que la media de los residuos es cero.")
    else:
        st.write("No hay suficiente evidencia para rechazar la hipótesis nula de que la media de los residuos es cero. Se cumple el supuesto.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3 style='text-align: center;'>3. La Varianza del error aleatorio es constante</h3>", unsafe_allow_html=True)
    st.write("Test de Breusch-Pagan para determinar la Homocedasticidad de los residuos.")

    st.markdown("$H_0:$ La homocedasticidad está presente")
    st.markdown("$H_1:$ La homocedasticidad no está presente (es decir, existe heterocedasticidad)")

    # Test de Breusch-Pagan para homocedasticidad
    names = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
    test = sms.het_breuschpagan(residuals, model.model.exog)

    st.write(lzip(names, test))

    p_value_bp = test[1]

    if p_value_bp < 0.05:
        st.write("Hay evidencia para rechazar la hipótesis nula de que existe homocedasticidad.")
    else:
        st.write("No hay suficiente evidencia para rechazar la hipótesis nula de que existe homocedasticidad. Se cumple el supuesto.")

with col2:
    st.markdown("<h3 style='text-align: center;'>4. Los errores además son idénticamente distribuidos y siguen distribución normal con media cero y varianza constante</h3>", unsafe_allow_html=True)   
    st.write("Test de Shapiro-Wilk (n < 30) o Normality Test (n >= 30).")

    st.markdown("$H_0:$ Los datos siguen una distribución Normal")
    st.markdown("$H_1:$ Los datos no siguen una distribución Normal")

    # Test de normalidad para los residuos
    _, norm_pvalue = normaltest(residuals)

    st.write(f'Normality Test p-value: {norm_pvalue}')

    if norm_pvalue < 0.05:
        st.write("Hay evidencia para rechazar la hipótesis nula de que los residuos siguen una distribución normal.")
    else:
        st.write("No hay suficiente evidencia para rechazar la hipótesis nula de que los residuos siguen una distribución normal. Se cumple el supuesto.") 

col1, col2 = st.columns(2)

with col1: 
    # Gráfico de residuos vs valores ajustados
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=model.fittedvalues, y=residuals, ax=ax)
    ax.axhline(0, ls='--', color='red')
    ax.set_xlabel('Valores Ajustados')
    ax.set_ylabel('Residuos')
    ax.set_title('Residuos vs. Valores Ajustados')
    st.pyplot(fig)

with col2:
    # Histograma de los residuos
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_xlabel('Residuos')
    ax.set_ylabel('Frecuencia')
    ax.set_title('Histograma de Residuos')
    st.pyplot(fig)

st.write("""
    Interpretación del análisis de los supuestos:

    - **Independencia de los Errores (Test de Durbin-Watson)**: Este test evalúa si los residuos están correlacionados. Un valor cercano a 2 sugiere que los residuos no están correlacionados, lo que indica que el supuesto se cumple.
    - **Media de los Errores es Cero**: Utilizamos un test para la media de una población para verificar que la media de los residuos es cero. Si no hay suficiente evidencia para rechazar la hipótesis nula, el supuesto se cumple.
    - **Homoscedasticidad (Test de Breusch-Pagan)**: Este test evalúa si la varianza de los errores es constante a través de los valores ajustados. Si no hay suficiente evidencia para rechazar la hipótesis nula, el supuesto de homocedasticidad se cumple.
    - **Normalidad de los Errores (Test de Normalidad)**: Utilizamos el test de normalidad para verificar si los residuos siguen una distribución normal. Si no hay suficiente evidencia para rechazar la hipótesis nula, el supuesto de normalidad se cumple.    
""")


# Convertir Productivity_Change a categorías numéricas
impact['Productivity_Change'] = impact['Productivity_Change'].map({'Increase': 1, 'No Change': 0, 'Decrease': -1})

# Rellenar NaNs con la media de las columnas relevantes para evitar dimensiones negativas
impact['Hours_Worked_Per_Week'].fillna(impact['Hours_Worked_Per_Week'].mean(), inplace=True)
impact['Productivity_Change'].fillna(impact['Productivity_Change'].mean(), inplace=True)

# Eliminar filas con cualquier NaN restante
impact = impact.dropna(subset=['Hours_Worked_Per_Week', 'Productivity_Change'])

# Ajustar el modelo ANOVA
modelo_anova = ols('Hours_Worked_Per_Week ~ C(Productivity_Change)', data=impact).fit()

# Realizar el análisis de varianza (ANOVA)
tabla_anova = sm.stats.anova_lm(modelo_anova, typ=2)

# Mostrar la tabla ANOVA en Streamlit
st.markdown("<h2 style='text-align: center;'>Estudio ANOVA de Cambio de Productividad</h2>", unsafe_allow_html=True)   
st.write(tabla_anova)

# Interpretación de la Tabla ANOVA
st.markdown("<h3 style='text-align: center;'>Interpretación de la Tabla ANOVA</h3>", unsafe_allow_html=True)   
st.write("""
La tabla ANOVA se interpreta de la siguiente manera:

- **sum_sq**: Esta es la suma de cuadrados. Para `Productivity_Change`, es la variabilidad explicada por `Productivity_Change`. Para `Residual`, es la variabilidad no explicada por `Productivity_Change`.
- **df**: Este es el grado de libertad. Para `Productivity_Change`, es el número de grupos menos 1. Para `Residual`, es el número total de observaciones menos el número de grupos.
- **F**: Este es el valor F, que es la razón entre la variabilidad explicada por `Productivity_Change` y la variabilidad no explicada por `Productivity_Change`. Un valor F grande sugiere que al menos uno de los grupos es significativamente diferente de los otros.
- **PR(>F)**: Este es el valor p, que es la probabilidad de obtener un valor F tan grande o mayor si la hipótesis nula es verdadera (es decir, si todos los grupos son realmente iguales). Un valor p pequeño sugiere que se puede rechazar la hipótesis nula.
""")

col1, col2 = st.columns(2)

with col1:
    # Boxplot para visualizar las diferencias entre los grupos
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Productivity_Change', y='Hours_Worked_Per_Week', data=impact, ax=ax, palette="Set3")
    ax.set_xlabel('Cambio en Productividad')
    ax.set_ylabel('Horas Trabajadas por Semana')
    ax.set_title('Boxplot: Cambio en Productividad vs Horas Trabajadas por Semana')
    st.pyplot(fig)

with col2:
    # Violinplot para visualizar la distribución de los datos en cada grupo
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x='Productivity_Change', y='Hours_Worked_Per_Week', data=impact, ax=ax, palette="Set3")
    ax.set_xlabel('Cambio en Productividad')
    ax.set_ylabel('Horas Trabajadas por Semana')
    ax.set_title('Violinplot: Cambio en Productividad vs Horas Trabajadas por Semana')
    st.pyplot(fig)

st.markdown('<h2 style="text-align: center;">Análisis de supuestos para ANOVA</h2>', unsafe_allow_html=True)

# Test de Anderson-Darling para cada grupo
groups = impact.groupby('Productivity_Change')
result_dict = {}

for group_name, group_data in groups:
    result = stats.anderson(group_data['Hours_Worked_Per_Week'])
    result_dict[group_name] = result

group_names = list(result_dict.keys())
cols = st.columns(len(groups))

# Mostrar resultados del Test de Anderson-Darling
for idx, group_name in enumerate(result_dict):
    result = result_dict[group_name]
    with cols[idx]:
        st.subheader(f"Grupo {group_name} - Test de Anderson-Darling")
        st.write(f'Estadístico: {result.statistic:.3f}')
        for i in range(len(result.critical_values)):
            sl, cv = result.significance_level[i], result.critical_values[i]
            if result.statistic < cv:
                st.write(f'Probablemente normal a nivel de significancia {sl:.1f}%')
            else:
                st.write(f'Probablemente no normal a nivel de significancia {sl:.1f}%')

# Test de Levene para homocedasticidad
group_data = [group['Hours_Worked_Per_Week'] for name, group in groups]
stat, p_value = stats.levene(*group_data)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Test de Levene para Homocedasticidad")
    st.write(f'Estadístico: {stat:.3f}, p-value: {p_value:.3f}')
    if p_value < 0.05:
        st.write("Hay evidencia para rechazar la hipótesis nula de que las varianzas son iguales. No se cumple el supuesto de homocedasticidad.")
    else:
        st.write("No hay suficiente evidencia para rechazar la hipótesis nula de que las varianzas son iguales. Se cumple el supuesto de homocedasticidad.")

    # Test de Durbin-Watson para independencia de los errores
    dw_stat = durbin_watson(modelo_anova.resid)

with col2:
    st.subheader("Test de Durbin-Watson para Independencia de los Errores")
    st.write(f'Estadístico de Durbin-Watson: {dw_stat:.3f}')
    alpha = 0.5
    if 2 - alpha <= dw_stat <= 2 + alpha:
        st.write("Los residuos no están correlacionados. Se cumple el supuesto de independencia.")
    elif dw_stat > 2 + alpha:
        st.write("Hay una autocorrelación negativa en los residuos.")
    else:
        st.write("Hay una autocorrelación positiva en los residuos.")
        
# Seleccionar las columnas numéricas y normalizarlas
numeric_cols = ['Age', 'Years_of_Experience', 'Hours_Worked_Per_Week', 'Number_of_Virtual_Meetings', 'Work_Life_Balance_Rating',
                'Social_Isolation_Rating', 'Company_Support_for_Remote_Work']

df_numeric = impact[numeric_cols]
df_normalize = StandardScaler().fit_transform(df_numeric)

# Realizar PCA
component_count = 5
pca = PCA(component_count)
principalComponents = pca.fit_transform(df_normalize)
principal_Df = pd.DataFrame(data=principalComponents, columns=[f'PC{i}' for i in range(1, component_count+1)])
variance_ratio = pca.explained_variance_ratio_

# Análisis de componentes
analysis = pd.DataFrame(pca.components_, columns=numeric_cols, index=[f'PC{i}' for i in range(1, component_count+1)]).T

# Streamlit app
st.markdown('<h2 style="text-align: center;">Análisis de Componentes Principales (PCA)</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.dataframe(analysis)
    
with col2:
    # Heatmap de análisis de componentes
    fig2, ax2 = plt.subplots()
    sns.heatmap(analysis, vmin=-1, vmax=1, cmap='coolwarm', ax=ax2)
    ax2.set_title('Heatmap de Análisis de Componentes')
    st.pyplot(fig2)


col1, col2 = st.columns(2)

with col1:
    st.bar_chart(principal_Df)

with col2:
    fig, ax = plt.subplots()
    ax.bar(np.arange(variance_ratio.shape[0]) + 1, variance_ratio)
    ax.set_title("Porciento de la varianza explicado por cada componente principal")
    ax.set_xlabel("Componentes principales")
    ax.set_ylabel("Porcentaje de varianza explicada")
    st.pyplot(fig)