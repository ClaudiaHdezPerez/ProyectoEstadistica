import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import geopandas as gpd
import scipy.stats as stats
import matplotlib.pyplot as plt

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
st.markdown("<h2 style='text-align: center;'>Análisis de las horas trabajadas por semana</h2>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<br>", unsafe_allow_html=True)
    men_hours = impact[impact['Gender'] == 'Male']['Hours_Worked_Per_Week']
    k2_men, p_men = stats.normaltest(men_hours)
    st.write(f'Estadístico K2 (Hombres): {k2_men}')
    st.write(f'Valor p (Hombres): {p_men}')
    if p_men > alpha:
        st.write("No se puede rechazar la hipótesis nula para hombres: los datos parecen provenir de una distribución normal.")
    else:
        st.write("Se rechaza la hipótesis nula para hombres: los datos no parecen provenir de una distribución normal.")

    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    
    non_men_hours = impact[impact['Gender'] != 'Male']['Hours_Worked_Per_Week']
    k2_non_men, p_non_men = stats.normaltest(non_men_hours)
    st.write(f'Estadístico K2 (No Hombres): {k2_non_men}')
    st.write(f'Valor p (No Hombres): {p_non_men}')
    if p_non_men > alpha:
        st.write("No se puede rechazar la hipótesis nula para no hombres: los datos parecen provenir de una distribución normal.")
    else:
        st.write("Se rechaza la hipótesis nula para no hombres: los datos no parecen provenir de una distribución normal.")

with col2:
    # Histograma y curva de distribución para hombres
    fig1, ax1 = plt.subplots()
    ax1.hist(men_hours, bins=30, edgecolor='black', density=True, label='Hombres')
    xmin, xmax = ax1.get_xlim()
    mean, std_dev = np.mean(men_hours), np.std(men_hours)
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean, std_dev)
    ax1.plot(x, p, 'k', linewidth=2, label='Curva Normal - Hombres')
    ax1.set_xlabel('Horas Trabajadas por Semana')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Histograma y Curva Normal para Hombres')
    ax1.legend()
    st.pyplot(fig1)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Histograma y curva de distribución para no hombres
    fig2, ax2 = plt.subplots()
    ax2.hist(non_men_hours, bins=30, edgecolor='black', density=True, label='No Hombres')
    xmin, xmax = ax2.get_xlim()
    mean, std_dev = np.mean(non_men_hours), np.std(non_men_hours)
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean, std_dev)
    ax2.plot(x, p, 'k', linewidth=2, label='Curva Normal - No Hombres')
    ax2.set_xlabel('Horas Trabajadas por Semana')
    ax2.set_ylabel('Frecuencia')
    ax2.set_title('Histograma y Curva Normal para No Hombres')
    ax2.legend()
    st.pyplot(fig2)

with col3:
    # Gráfico Q-Q para hombres
    fig3, ax3 = plt.subplots()
    stats.probplot(men_hours, dist="norm", plot=ax3)
    ax3.set_title('Gráfico Q-Q para Hombres')
    st.pyplot(fig3)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Gráfico Q-Q para no hombres
    fig4, ax4 = plt.subplots()
    stats.probplot(non_men_hours, dist="norm", plot=ax4)
    ax4.set_title('Gráfico Q-Q para No Hombres')
    st.pyplot(fig4)
    

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

# Pair plot
st.markdown('#### Pair Plot')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Ejemplo de datos (asegúrate de que men_hours y non_men_hours estén definidos)
min_length = min(len(men_hours.values), len(non_men_hours.values))
data = {
    'Hombres': men_hours.values.tolist()[:min_length],
    'Resto de Géneros': non_men_hours.values.tolist()[:min_length],
    'Grupo': ['Hombres'] * 635 + ['Resto de Géneros'] * 635
}
df = pd.DataFrame(data)

# Crear el pair plot
fig = sns.pairplot(df, hue='Grupo').fig

# Mostrar el gráfico en Streamlit
st.pyplot(fig)

st.markdown('''
La prueba de hipótesis indica que los hombres tienden a trabajar más horas de media a la semana que otros 
géneros en esta muestra, subrayando la necesidad de abordar los posibles riesgos que este comportamiento 
puede tener en su salud mental. Es fundamental que tanto los empleadores como los empleados consideren el 
equilibrio entre el trabajo y la vida personal como una prioridad, para prevenir el estrés crónico, el 
agotamiento y otros problemas relacionados con la salud mental. A largo plazo, promover una cultura laboral 
que valore el bienestar de los empleados puede ser esencial para mejorar la salud mental de los hombres y 
de la fuerza laboral en general.           
''')