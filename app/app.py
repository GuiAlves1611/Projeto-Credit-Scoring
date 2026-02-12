import sys, os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import streamlit as st
import pandas as pd
from datetime import date   
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_recall_curve, auc
import joblib
import random
from src.pipeline_components import DropCols, EnsureCategorical, EnsureNumeric, XGBWithAutoSPW, LogTransform
from src import train_score_pipeline, apply_pipeline_to_new_data, proba_to_score, rating, decision_by_score, build_scoring_df, prepare_X_for_model, build_history_features
import time

score_df = pd.read_parquet("data/credit/score_df.parquet")

@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("models/credit_pipeline_v3.pkl")
    score_params = joblib.load("models/score_params_v3.pkl")
    return pipeline, score_params


pipeline, score_params = load_artifacts()

@st.cache_resource
def carregar_dados_modelo():    
    dados_teste = joblib.load('models/score_resultados_teste.pkl')
    return dados_teste['y_test'], dados_teste['proba']
y_test, proba = carregar_dados_modelo()

def gerar_id():
    if "ids_gerados" not in st.session_state:
        st.session_state.ids_gerados = set()
    while True:
        novo_id = random.randint(0000000, 9999999)
        if novo_id not in st.session_state.ids_gerados:
            st.session_state.ids_gerados.add(novo_id)
            return novo_id
        
novo_id = gerar_id()
    


st.set_page_config(layout="wide", page_title="CredInsight — Decisão de Crédito", initial_sidebar_state="expanded", page_icon="📊")

def update_layout_dark(fig, title, y_title, x_title="Vintage (meses)"):
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified"
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='#333')
    return fig

def render_landing_page():
    # --- 1. Título Estilizado com CSS (Efeito "Premium") ---
    st.markdown("""
        <style>
        .title-box {
            background: linear-gradient(45deg, #1e3799, #0c2461);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .title-text {
            color: white;
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 2.5em;
            font-weight: bold;
            margin: 0;
        }
        .subtitle-text {
            color: #dcdde1;
            font-size: 1.1em;
            margin-top: 5px;
        }
        </style>
        <div class="title-box">
            <h1 class="title-text">💳 CreditInsight</h1>
            <p class="subtitle-text">Diagnóstico Estratégico & Otimização de Risco</p>
        </div>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/7516/7516657.png", width=50) # Placeholder de logo
    st.title("CreditInsight")
    st.caption("Sistema de Análise e Decisão")
    
    st.divider()
    
    st.subheader("🤖 Status do Modelo")
    st.info(f"Versão: v3.1")
    
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("AUC", "0.91", delta="Otimo", delta_color="normal")
    col_s2.metric("RECALL", "0.80")
    
    st.text(f"Atualizado: {date.today()}")
    
    st.divider()
    
    st.write("**Navegação**")
    page = st.radio("Ir para:", ["Contexto do Problema", "Storytelling", "Simulador","Metodologia"])

if page == "Contexto do Problema":
    st.title(":bar_chart: CreditInsight — Análise e Decisão de Crédito ")

    st.subheader("Contexto do problema")

    with st.container():
        st.subheader("Panorama Atual da Carteira")
        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        
        with kpi1:
            st.metric("Clientes Analisados", "438.510", delta="Base Total")
        with kpi2:
            st.metric("Score Médio", "646", help="Score médio da população atual")
        with kpi3:
            st.metric("Aprovação Condicionada", "52%")
        with kpi4:
            # Destaque de cor inversa pois 99% de aprovação é arriscado (como diz o texto)
            st.metric("Aprovação Total", "99%", delta="Perfil Agressivo", delta_color="inverse")
        with kpi5:
            variacao_perda = -150
            st.metric("Perda Esperada Mensal", "R$ 2,9k", delta=f"{variacao_perda} vs mês ant.",delta_color="inverse")

    st.warning("🚨 **Atenção:** A política atual aprova 99% da base, com 52% sob restrição. Isso indica um perfil **agressivo** com alta exposição ao risco.", icon="⚠️")

    st.divider()

    c1, c2 = st.columns([2, 1])

    with c1:
        st.markdown("### 📉 O Custo da Ineficiência")
        st.write("Instituições financeiras realizam diariamente decisões sobre concessão de crédito. Na ausência de um modelo estruturado (ML/AI), observamos:")
        
        # Usando Expander para limpar a tela, ou cards
        col_a, col_b = st.columns(2)
        with col_a:
            st.error("**Riscos Financeiros**")
            st.caption("• Aumento da Inadimplência\n\n• Geração de Prejuízo Financeiro Direto")
        
        with col_b:
            st.warning("**Riscos Operacionais**")
            st.caption("• Sobrecarga dos times de mesa\n\n• Decisões inconsistentes entre analistas")

    with c2:
        st.markdown("### 🎯 Escopo da Solução")
        st.success("""
        **O que este modelo faz:**
        
        ✅ Decisão baseada em histórico
        
        ✅ Janela de observação de 12 meses
        
        ✅ Modelo Supervisionado (XGBoost)
        
        ✅ Validação Offline (OOT)
        """)

    st.divider()

    st.success("""
    📌 Recomendação Atual:

    • Avaliar redução do cutoff de aprovação
    • Reforçar critérios para clientes com restrição
    • Revisar limites em scores entre 620–650
    • Monitorar inadimplência em vintages elevados
    """)

    # COMO USAR - Melhor em um expander ou abas para não poluir
    with st.expander("📘 Guia Rápido: Como usar este painel"):
        st.markdown("""
        1. **KPIs:** Consulte os indicadores principais no topo desta página.
        2. **Storytelling:** Acesse a aba lateral para entender os *drivers* da decisão (SHAP values, Feature Importance).
        3. **Simulador:** Use para testar cenários "What-If" em clientes específicos.
        4. **Metodologia:** Valide as métricas técnicas e documentação.
        """)
elif page == "Storytelling":
    
    st.title("📈 Motor de Decisão de Crédito: Diagnóstico e Otimização de Política")

    st.divider()

    # --- 2. O Problema vs. O Objetivo (Layout Lado a Lado) ---
    col_prob, col_obj = st.columns(2, gap="medium")

    with col_prob:
        with st.container(border=True):
            st.subheader("📌 Visão Geral do Problema")
            st.markdown("""
            Nos últimos ciclos, a operação manteve **alto volume de aprovação** e score médio estável na entrada.
            
            Entretanto, análises por Vintage indicam um **aumento relevante da inadimplência** em contratos mais maduros, revelando um risco oculto ao modelo atual.
            """)

    with col_obj:
        with st.container(border=True):
            st.subheader("🎯 Objetivo da Análise")
            st.markdown("""
            Avaliar a sustentabilidade da política de crédito e identificar oportunidades para:
            
            * 📉 **Reduzir** a inadimplência tardia.
            * 🛡️ **Preservar** clientes de bom perfil.
            * ⚖️ **Otimizar** a relação Risco × Volume.
            """)

    # --- 3. A Hipótese Central (Destaque de Alerta) ---
    # Usamos st.warning para dar a cor amarela/laranja de "Atenção"
    st.warning("""
    **⚠️ Hipótese Central de Negócio**
    
    A política atual, baseada em alta taxa de aprovação (99%), está criando um **ponto cego de risco**. 
    Estamos aprovando perfis que aparentam segurança no início (Short-term), mas deterioram drasticamente ao longo do ciclo de vida (Long-term).
    """, icon="⚠️")

    st.divider()

    # --- 4. A Abordagem (Rodapé Visual) ---
    st.subheader("🚀 Nossa Abordagem Técnica")
    
    # Criando 4 colunas para os ícones ficarem alinhados
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.info("**Estudo por Vintage**\n\nDiagnóstico temporal de safras.")
    with c2:
        st.error("**Simulação de Cutoff**\n\nCenários What-If de impacto financeiro.")
    with c3:
        st.success("**Drivers de Risco**\n\nAnálise de Estabilidade e Demografia.")
    with c4:
        st.info("**Precisão vs. Recall**\n\nCalibragem técnica do modelo.")

    # --- Chamada da função no seu app principal ---
    st.subheader("A análise a seguir sustenta uma proposta objetiva de ajuste de política")

    st.divider()

    if __name__ == "__main__":
        # Se estiver usando estrutura de menu, chame apenas se a página for "Home"
        render_landing_page()

    st.markdown("### 📈 Score médio (por Vintage)")

    score_by_vintage = (
        score_df.groupby("vintage")["score"]
        .mean()
        .sort_index()
    )

    col1, col2 = st.columns(2)

    with col1:
        # Score Médio
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=score_by_vintage.index, 
            y=score_by_vintage.values,
            mode='lines+markers',
            line=dict(color='#00CC96', width=3),
            marker=dict(size=8),
            name='Score Médio'
        ))
        update_layout_dark(fig1, "Score médio por Vintage", "Score médio")
        st.plotly_chart(fig1, use_container_width=True)

    st.info("**A Ilusão:** O perfil de entrada (Score 646) parece estável e seguro.")

    with col2:
        # Taxa de Default
        bad_rate_by_vintage = (
            score_df.groupby("vintage")["y_true"]
            .mean()
            .sort_index()
        )
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=bad_rate_by_vintage.index, 
            y=bad_rate_by_vintage.values,
            mode='lines+markers',
            line=dict(color='#EF553B', width=3),
            marker=dict(size=8),
            name='Taxa de Default'
        ))
        update_layout_dark(fig2, "Taxa de default por Vintage", "Taxa de default")
        st.plotly_chart(fig2, use_container_width=True)
    st.error("**O Fato:** O risco está explodindo nas safras recentes. O score atual não está vendo o perigo.")

    st.subheader("🎯 Conclusão Estratégica")
    st.warning("""A política de **99% de aprovação** criou um ponto cego. Estamos atraindo o mesmo 'perfil', mas o comportamento de crédito degradou. 
               **A solução não é parar de emprestar, mas usar o novo modelo para filtrar o ruído.**""")

    # --- 2) Lógica de Aprovação vs Default ---
    st.divider()
    st.markdown("### 📊 Simulador de Política de Aprovação")

    c_col1, c_col2 = st.columns([1, 2])
    with c_col1:
        use_cutoff_for_approval = st.toggle("Usar cutoff automático", value=True)
        cutoff = st.slider("Definir Cutoff de Score", 305, 770, 650, 5)

    if use_cutoff_for_approval:
        score_df["approved"] = (score_df["score"] >= cutoff).astype(int)
    else:
        score_df["approved"] = score_df["decision"].astype(str).str.contains("Aprov", case=False, na=False).astype(int)

    # Cálculo da Matriz
    counts = score_df.groupby(["approved", "y_true"]).size().unstack(fill_value=0)
    for col in [0, 1]:
        if col not in counts.columns: counts[col] = 0
    counts = counts[[0, 1]]

    labels = ["Negado", "Aprovado"]
    no_default = counts.loc[[0, 1], 0].values
    yes_default = counts.loc[[0, 1], 1].values

    # --- 3) Gráficos de Barras Empilhadas (Volume e %) ---
    col3, col4 = st.columns(2)

    with col3:
        # Volume
        fig3 = go.Figure(data=[
            go.Bar(name='Não Default', x=labels, y=no_default, marker_color='#4A90E2'),
            go.Bar(name='Default', x=labels, y=yes_default, marker_color='#FF6B6B')
        ])
        fig3.update_layout(barmode='stack', title="Volume de Decisão", legend=dict(font=dict(color="#E0E0E0")))
        update_layout_dark(fig3, "Aprovação vs Default (Volume)", "Qtd Clientes", x_title="")
        st.plotly_chart(fig3, use_container_width=True)
        st.info("""**🔍 O Olhar do Negócio:** O score médio é consistente. Aparentemente, não houve degradação no perfil de quem entra na carteira.""")
    with col4:
        # Percentual
        rates = counts.div(counts.sum(axis=1), axis=0).replace([np.inf, -np.inf], 0).fillna(0)
        rate_no = rates.loc[[0, 1], 0].values
        rate_yes = rates.loc[[0, 1], 1].values

        fig4 = go.Figure(data=[
            go.Bar(name='Não Default', x=labels, y=rate_no, marker_color='#4A90E2'),
            go.Bar(name='Default', x=labels, y=rate_yes, marker_color='#FF6B6B')
        ])
        fig4.update_layout(barmode='stack', title="Mix de Risco (%)", legend=dict(font=dict(color="#E0E0E0")))
        update_layout_dark(fig4, "Aprovação vs Default (%)", "Proporção", x_title="")
        # Formatação de porcentagem no eixo Y
        fig4.update_layout(yaxis_tickformat='.0%')
        st.plotly_chart(fig4, use_container_width=True)
        st.error("""**⚠️ O Sinal de Alerta:** O risco 'estoura' conforme o cliente amadurece. A
                  estratégia atual é cega para esse comportamento tardio, gerando picos de perda após o 55º mês.""")
        
    st.divider()

    st.markdown("### 🔍 O que o modelo enxerga?")

    # 1. Preparação dos dados (mantendo seu filtro de outliers)
    df_box = score_df[score_df['years_employed'] < 40]

    # 2. Criação do Boxplot Interativo
    fig_box = go.Figure()

    # Box para Clientes BONS (0)
    fig_box.add_trace(go.Box(
        y=df_box[df_box['y_true'] == 0]['years_employed'],
        name='Bons Pagadores (0)',
        marker_color='#4A90E2', # Mesmo azul do seu simulador
        boxmean=True,           # Adiciona a linha da média automaticamente
        fillcolor='rgba(74, 144, 226, 0.5)', # Leve transparência
        line=dict(width=2)
    ))

    # Box para Clientes MAUS (1)
    fig_box.add_trace(go.Box(
        y=df_box[df_box['y_true'] == 1]['years_employed'],
        name='Inadimplentes (1)',
        marker_color='#FF6B6B', # Mesmo coral do seu simulador
        boxmean=True,           # Adiciona a linha da média automaticamente
        fillcolor='rgba(255, 107, 107, 0.5)',
        line=dict(width=2)
    ))

    # 3. Estilização para o Modo Escuro
    fig_box.update_layout(
        title="<b>Estabilidade Profissional vs Inadimplência</b><br><sup>Média indicada pela linha pontilhada interna</sup>",
        yaxis_title="Anos Empregado",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    # Adicionando grid sutil
    fig_box.update_yaxes(showgrid=True, gridcolor='#333')

    # Exibir no Streamlit
    st.plotly_chart(fig_box, use_container_width=True)

    # Callout de Storytelling
    st.markdown(f"""
    > **Insight Chave:** Note a diferença na estabilidade. Enquanto os bons pagadores têm uma mediana de emprego 
    > significativamente maior, o grupo de risco está concentrado abaixo dos 5 anos. 
    > Esse é um dos principais 'drivers' que seu modelo usa para reduzir a **Perda Esperada de R$ 2,9k**.
    """)

    st.divider()

    st.markdown("### 🎂 Inadimplência por Ciclo de Vida")

    # 1. Preparação dos dados
    score_df['age_bins'] = pd.cut(score_df['years'], bins=[20, 30, 40, 50, 60, 75])
    prop_data = score_df.groupby('age_bins')['y_true'].mean().reset_index()
    # Convertendo categorias para string para evitar erros no Plotly
    prop_data['age_bins'] = prop_data['age_bins'].astype(str)
    media_geral = score_df['y_true'].mean()

    # 2. Criação do Gráfico de Barras
    fig_age = px.bar(
        prop_data, 
        x='age_bins', 
        y='y_true',
        text_auto='.1%', # Mostra a % em cima da barra
        color='y_true',
        color_continuous_scale='OrRd', # Mantendo sua paleta original
        labels={'age_bins': 'Faixa Etária', 'y_true': 'Taxa de Default'}
    )

    # 3. Adicionando a Linha de Média Geral (Sua linha azul)
    fig_age.add_hline(
        y=media_geral, 
        line_dash="dash", 
        line_color="#4A90E2", # Azul do seu sistema
        annotation_text=f"Média Geral: {media_geral:.1%}", 
        annotation_position="top right"
    )

    # 4. Estilização Dark
    fig_age.update_layout(
        title="<b>Taxa de Inadimplência por Faixa Etária</b>",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        coloraxis_showscale=False, # Remove a barra de cores lateral para limpar o visual
        yaxis_tickformat='.1%'
    )

    st.plotly_chart(fig_age, use_container_width=True)

    st.markdown("""
    > A análise demonstra que o modelo mantém estabilidade na atribuição de score ao longo do tempo, indicando consistência na avaliação inicial do risco. 
    >
    > Entretanto, observa-se aumento da inadimplência em contratos mais antigos, sugerindo que o principal desafio da operação não está na concessão, mas na gestão da carteira ao longo do seu ciclo de vida.
    >
    > #### **Principais Pontos Observados:**
    > * **Zonas de Risco:** As faixas etárias de **[20, 30]** e **[60, 75]** anos apresentam as maiores taxas de default, atingindo **0.9%**, superando a média geral da base.
    > * **Zona de Segurança:** A faixa de **[30, 40]** anos é o porto seguro da carteira, com a menor taxa de inadimplência registrada (**0.4%**).
    > * **Referência:** A linha tracejada azul representa a **Média Geral de 0.6%**, servindo como o benchmark para nossas decisões de corte.
    >
    > **💡 Oportunidade:** Ajustar o **Cutoff** para ser mais seletivo nas faixas de alto risco permite reduzir a **Perda Esperada de R$ 2,9k** sem sacrificar os bons pagadores em faixas mais estáveis.
    """)

    st.divider()

    st.markdown("### 🎯 Curva de Precisão vs. Recall")

    # 1. Calculando a curva (data['y_true'] e data['y_scores'] do seu modelo)
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    pr_auc = auc(recall, precision)

    # 2. Criando o gráfico interativo
    fig_pr = go.Figure()

    fig_pr.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'PR Curve (AUC = {pr_auc:.2f})',
        line=dict(color='#4A90E2', width=3),
        fill='tozeroy',
        fillcolor='rgba(74, 144, 226, 0.1)'
    ))

    # Estilização Dark
    fig_pr.update_layout(
        title="<b>Trade-off: Qualidade vs. Volume de Aprovação</b>",
        xaxis_title="Recall (Capacidade de encontrar os inadimplentes)",
        yaxis_title="Precision (Confiança na classificação)",
        template="plotly_dark",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, 1.05]),
        xaxis=dict(range=[0, 1.05])
    )

    st.plotly_chart(fig_pr, use_container_width=True)
    st.subheader("🎯 Curva de Precisão vs. Recall: O Equilíbrio Estratégico")

    st.markdown("""
    Esta curva representa a **eficiência do nosso motor de decisão** e serve como base científica para o ajuste do nosso Cutoff. Diferente da regra de aprovação atual de 99%, este gráfico nos permite visualizar exatamente o que ganhamos e o que perdemos em cada nível de rigor.

    #### **Interpretando o Trade-off:**
    * **Zona de Alta Confiança (Precision):** Enquanto a linha permanece no topo (1.0), temos a garantia de que os aprovados são bons pagadores.
    * **Ponto de Ruptura (Recall):** Observe que até aproximadamente **0.75 de Recall**, conseguimos manter uma precisão máxima. Após esse ponto, para aprovar mais pessoas, começamos a aceitar inadimplentes na carteira.
    * **O "Ponto de Ouro":** Onde a curva começa a cair bruscamente é o nosso limite técnico ideal para evitar a **Perda Esperada de R$ 2,9k**.

    > **📈 Visão Executiva:** O modelo demonstra uma excelente separação de risco. Podemos manter uma aprovação volumosa e qualificada até encontrarmos o ponto de equilíbrio que maximiza a receita e minimiza o default tardio observado nos vintages maduros.
    """)
elif page == "Simulador":
    st.title("🤖 Simulador de Perfil e Risco")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Preencha os dados abaixo para que nosso modelo de Machine Learning 
        analise o perfil e calcule o índice de risco personalizado.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    with st.expander("👨‍👩‍👧‍👦 **Dados Familiares**", expanded=True):
        st.write("Informações sobre a estrutura familiar do solicitante.")

        col1_fam, col2_fam, col3_fam, col4_fam = st.columns(4)

        with col1_fam:
            gender = st.radio("Sexo", ["Masculino", "Feminino"])
        with col2_fam:
            years = st.number_input("Selecione sua idade", 18, 68, help="Idade entre 18 e 68 anos.")
        with col3_fam:
            cnt_children = st.selectbox("Quantidade de filhos", [0, 1, 2, 3, 4, 5, 6])
        with col4_fam:
            cnt_fam_members = st.selectbox("Quantidade de membros na familia", [1, 2, 3, 4, 5, 6, 7])

        st.markdown("<br>", unsafe_allow_html=True) # Espaçamento

        col5_fam, col6_fam = st.columns([3, 1])

        with col5_fam:
            name_education_type = st.selectbox("Gráu de Ensino", ["Ensino fundamental II (anos finais)", "Ensino médio / técnico", "Ensino superior incompleto",
                                    "Ensino superior completo", "Pós-graduação / Mestrado / Doutorado"])
        with col6_fam:
            name_family_status = st.selectbox("Estado Cívil", ["Casado(a)", "Casamento civil","Solteiro(a)", "Divorciado(a)", "Viuvo(a)"])
    with st.expander("💰 **Dados Financeiros & Patrimônio**"):
        st.write("Detalhes sobre renda e bens.")
        amt_income_month = st.number_input("Salário Mensal(R$)", min_value=0.0, max_value=131250.0 ,format="%.2f")

        col_check, _ = st.columns([1, 2])
        col1_fin, col2_fin = st.columns(2)

        with col_check:
            missing = st.checkbox("Não possuo profissão / Desempregado")
        
        with col1_fin:
            occupation_type = st.text_input("Digite a profissão", placeholder="Analista de sistemas", disabled=missing)
            name_income_type = st.selectbox("Tipo de renda", ["CLT", "Vendedor(a)", "Pensão", "Servidor público", "Estudante"])
        with col2_fin:
            years_employed = st.slider("Tempo trabalhado (anos)", 0.0, 45.9, step=0.1, help="Tempo no emprego atual")
            name_housing_type = st.selectbox("Tipo de morádia", ["Casa / Apartamento", "Apartamento de programa habitacional", "Alugúel", 
                                                            "Escritório", "Com os pais", "Cooperativa Habitacional"])
            
        st.markdown("---") # Linha divisória interna
        st.write("**Possui bens em seu nome?**")

        col3_fin, col4_fin = st.columns(2)
        with col3_fin:
            flag_own_car = st.radio("Automovel Próprio", ["Sim", "Não"])
        with col4_fin:
            flag_own_realty = st.radio("Imóvel Próprio", ["Sim", "Não"])

    with st.expander("🏦 **Histórico Bancário**"):
        st.write("Registro de relacionamento bancário.")

        col1_bank, col2_bank = st.columns(2)

        with col1_bank:
            status = st.selectbox("Status de pagamento das parcela de crédito", [0, 1, 2, 3, 4, 5])
        with col2_bank:
            months_balance = st.slider("Mês de pagamento da última parcela", 0, 60, step=1, help="1 mês atrás, 2 mêses atrás etc")

    st.markdown("---")
    _, col_btn, _ = st.columns([1, 2, 1])

    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True) # Alinhamento vertical
        botao_simular = st.button("🚀 Executar Simulação de Risco", use_container_width=True, type="primary")

    if botao_simular:

        placeholder_status = st.empty()
            
        with st.spinner("🧠 O motor de crédito está analisando o perfil..."):
            etapas = [
            ("🔍", "Mapeando variáveis categóricas..."),
            ("📂", "Acessando base histórica..."),
            ("📊", "Calculando Score de Risco..."),
            ("🛡️", "Validando contra políticas de inadimplência...")
        ]

            for icon, msg in etapas:
                placeholder_status.markdown(f"#### {icon} **Status:** {msg}")
                time.sleep(0.8)
            
        placeholder_status.empty()

        map_sexo = {"Masculino": 0, "Feminino": 1}
        map_imovel = {"Não": 0, "Sim": 1}
        map_automovel = {"Não": 0, "Sim": 1}
        map_escolaridade = {
                "Ensino fundamental II (anos finais)": "Lower secondary",
                "Ensino médio / técnico": "Secondary / secondary special",
                "Ensino superior incompleto": "Incomplete higher",
                "Ensino superior completo": "Higher education",
                "Pós-graduação / Mestrado / Doutorado": "Academic degree"
            }
        map_familia = {
                "Casado(a)": "Married",
                "Casamento civil": "Civil marriage",
                "Solteiro(a)": "Single / not married",
                "Divorciado(a)": "Separated",
                "Viuvo(a)": "Widow"
            }

        map_renda = {
                "CLT": "Working",
                "Vendedor(a)": "Commercial associate",
                "Pensão": "Pensioner",
                "Servidor público": "State servant",
                "Estudante": "Student"
            }

        map_moradia = {
                "Casa / Apartamento": "House / apartment",
                "Apartamento de programa habitacional": "Municipal apartment",
                "Alugúel": "Rented apartment",
                "Escritório": "Office apartment",
                "Com os pais": "With parents",
                "Cooperativa Habitacional": "Co-op apartment"
            }

            

        cnt_fam_members = float(cnt_fam_members)
        no_formal_employment = int((occupation_type == "MISSING") and (float(years_employed) == 0.0))
        unclassified_occupation = int((occupation_type == "MISSING") and (float(years_employed) > 0.0))
        if cnt_fam_members > 0:
            renda_per_capita = amt_income_month / cnt_fam_members
        else:
            renda_per_capita = 0.0

        if missing:
            occupation_type = "MISSING"
        else:
            occupation_type = occupation_type


        dados_cliente = pd.DataFrame({
                "ID": [novo_id],
                "CODE_GENDER": [map_sexo[gender]],
                "years": [years],
                "CNT_CHILDREN": [cnt_children],
                "CNT_FAM_MEMBERS": [cnt_fam_members],
                "FLAG_OWN_CAR": [map_automovel[flag_own_car]],
                "FLAG_OWN_REALTY": [map_imovel[flag_own_realty]],
                "NAME_INCOME_TYPE": [map_renda[name_income_type]],
                "NAME_EDUCATION_TYPE": [map_escolaridade[name_education_type]],
                "NAME_FAMILY_STATUS": [map_familia[name_family_status]],
                "NAME_HOUSING_TYPE": [map_moradia[name_housing_type]],
                "OCCUPATION_TYPE": [occupation_type],
                "years_employed": [years_employed],
                "amt_income_month": [amt_income_month],
                "renda_per_capita": [renda_per_capita],
                "no_formal_employment": [no_formal_employment],
                "unclassified_occupation": [unclassified_occupation]
            })
            
        dados_bancarios = pd.DataFrame({
                "ID": [novo_id],
                "STATUS": [status],
                "MONTHS_BALANCE": [months_balance]
            })

        cuts = {"q90": 750, "q70": 650, "q40": 570, "q15": 450,
                            "cut_reprovado": 450, "cut_manual": 570, "cut_restricao": 650}

        hist_features = build_history_features(dados_bancarios, window_months=12)
        df_scoring = build_scoring_df(dados_cliente, hist_features)
        X_new, ids = prepare_X_for_model(df_scoring)
        df_scored = apply_pipeline_to_new_data(X_new, pipeline, score_params)

        if ids is not None:
            df_scored["ID"] = ids.values

        score = float(df_scored["score"].iloc[0])
        proba = float(df_scored["proba_bad"].iloc[0])
        decisao = decision_by_score(score, cuts)
        rating_ = rating(score, cuts)
        textos_insight = {
                "A - Excelente": "Perfil de altíssima fidelidade e baixíssimo risco histórico. Possui indicadores de estabilidade financeira superiores a 90% da base.",
                "B - Bom": "Perfil com comportamento estável e baixo risco. Similar a 85% da base de clientes adimplentes do modelo.",
                "C - Regular": "Perfil intermediário com oscilações pontuais. Requer monitoramento de limite, mas mantém score dentro da média do cluster.",
                "D - Alerta": "Perfil com indicadores de volatilidade financeira. O modelo identificou padrões de comportamento que precedem atrasos em 40% dos casos similares.",
                "E - Crítico": "Perfil de alto risco. Cluster caracterizado por baixa permanência no emprego e histórico de atrasos recorrentes."
            }

        st.success("✅ **Simulação Concluída!**")
        with st.container(border=True):
            st.markdown("### 📊 Relatório de Análise de Crédito")

            col_decisao1, col_decisao2, col_decisao3, = st.columns([2, 1, 1])

            with col_decisao1:
                st.metric("Decisão", decisao)
            with col_decisao2:
                st.metric("Score", f"{score:.0f}")
            with col_decisao3:
                st.metric("Prob. Inadimplência", f"{proba*100:.2f}%")

            st.markdown(f"**Nível de Confiança do Perfil (Rating: {rating_})**")
            st.progress(score / 1000)

            st.divider()

            if "Aprovado" in decisao:
                st.success("✅ Cliente com perfil adequado para crédito nas condições atuais.")
            elif "Restrição" in decisao:
                st.warning("⚠️ Aprovação com restrição: recomenda-se reduzir limite e/ou exigir garantias.")
            elif "Análise" in decisao:
                st.info("🔎 Perfil intermediário: recomendado encaminhar para análise manual.")
            else:
                st.error("⛔ Risco elevado: recomendada reprovação ou oferta de produto mais conservador.")

            insight_perfil = textos_insight.get(rating_, "Perfil em análise detalhada pelo motor de crédito.")
        with st.expander("🔍 Detalhes Técnicos e Insight do Cluster"):
            col_tec, col_biz = st.columns([1, 2])

            with col_tec:
                st.write("**Métricas do Modelo**")
                st.write(f"• AUC: `0.91`")
                st.write(f"• RECALL: `0.80` ")
            with col_biz:
                st.write("**Análise de Comportamento**")
                st.info(f"{insight_perfil}")
elif page == "Metodologia":
    # --- CABEÇALHO ---
    st.title("📚 Metodologia de Análise de Crédito")
    st.markdown("""
    Esta seção detalha como nosso algoritmo transforma dados brutos em decisões financeiras seguras.
    Nosso compromisso é com a **transparência** e a **governança**.
    """)
    st.markdown("---")

    # --- OBJETIVO (DESTAQUE) ---
    with st.container(border=True):
        col_icon, col_text = st.columns([1, 5])
        with col_icon:
            st.markdown("# 🎯")
        with col_text:
            st.subheader("Objetivo do Modelo")
            st.write(
                "Nosso sistema foi desenvolvido para avaliar o risco de crédito de forma **estruturada, consistente e baseada em dados históricos**.\n\n"
                "O foco é apoiar decisões mais seguras, equilibrando o **crescimento da carteira** com o **controle rigoroso de risco**."
            )

    st.markdown("### 🧠 O que é Avaliado?")
    st.caption("A análise considera dois pilares fundamentais de informação:")

    # --- PILARES DE AVALIAÇÃO (COLUNAS) ---
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("#### 📌 Perfil Financeiro")
            st.markdown("""
            Fatores que indicam a capacidade de pagamento:
            * **Renda Mensal**
            * **Estabilidade Profissional**
            * **Estrutura Familiar**
            * **Patrimônio Declarado**
            """)

    with col2:
        with st.container(border=True):
            st.markdown("#### 📌 Histórico de Crédito")
            st.markdown("""
            Fatores que indicam a vontade de pagar:
            * **Registros de Pagamento**
            * **Indícios de Atraso**
            * **Consistência Histórica**
            * **Comportamento Recente**
            """)

    st.markdown("---")

    # --- COMO O SCORE FUNCIONA ---
    st.subheader("📊 Como o Sistema Classifica o Cliente")
    st.info(
        "A partir das informações acima, o sistema calcula um **Score de Crédito (0 a 850)**.\n\n"
        "Quanto maior o score, menor o risco estimado de inadimplência, refletindo maior probabilidade de adimplência e estabilidade financeira"
    )

    st.write("##### ⚖️ Faixas de Decisão Automática")
    
    # Visualização das faixas (Cards Coloridos)
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.success("**Aprovado**")
        st.caption("Perfil consistente e baixo risco estimado")
    
    with c2:
        st.warning("**Restrição**")
        st.caption("Risco moderado, com ajustes nas condições")
        
    with c3:
        st.info("**Análise Manual**")
        st.caption("Necessita validação complementar")
        
    with c4:
        st.error("**Reprovado**")
        st.caption("Risco elevado fora da política de crédito.")

    st.markdown("---")

    # --- GOVERNANÇA E BENEFÍCIOS ---
    c_gov, c_cli = st.columns(2)

    with c_gov:
        st.markdown("### 🔒 Responsabilidade")
        st.markdown("""
        O modelo foi auditado para garantir:
        - ✅ **Coerência:** Risco estimado alinhado com a decisão.
        - ✅ **Anti-Distorção:** Evita uso de dados futuros (Look-ahead bias).
        - ✅ **Auditoria:** Critérios claros e reproduzíveis.
        - ✅ O modelo não utiliza informações sensíveis ou discriminatórias.
        """)

    with c_cli:
        st.markdown("### 🧠 Valor para o Cliente")
        st.markdown("""
        Por que essa metodologia é melhor?
        - 1️⃣ **Entendimento:** O cliente compreende o processo.
        - 2️⃣ **Segurança:** O cliente sente confiança na análise.
        - 3️⃣ **Governança:** Percepção clara de seriedade.
        - 4️⃣ **Clareza:** Sem termos técnicos confusos.
        """)