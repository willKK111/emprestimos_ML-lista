import streamlit as st
import pandas as pd
import joblib

# Configuração da página
st.set_page_config(
    page_title="Análise de Empréstimos",
    layout="centered"
)

# Carregar modelo e encoders
@st.cache_resource
def carregar_modelos():
    modelo = joblib.load("modelo_loan_status.pkl")
    try:
        encoders = joblib.load("encoders.pkl")
    except Exception:
        encoders = None
    return modelo, encoders

modelo, encoders = carregar_modelos()

# Título
st.title("Análise de Aprovação de Empréstimos")
st.write(
    "Preencha os dados do formulário abaixo para avaliar a solicitação de crédito."
)

st.divider()

# Formulário
with st.form("formulario_emprestimo"):

    st.subheader("Dados do Solicitante")

    # 1. Sexo (radio)
    gender = st.radio(
        "Sexo:",
        options=[0, 1],
        format_func=lambda x: "Feminino" if x == 0 else "Masculino",
        horizontal=True
    )

    # 2. Dependentes (selectbox/option)
    dependents = st.selectbox(
        "Número de dependentes:",
        options=[0, 1, 2, 3],
        format_func=lambda x: "3 ou mais" if x == 3 else str(x)
    )

    # 3. Casado (radio)
    married = st.radio(
        "Casado(a)?",
        options=[0, 1],
        format_func=lambda x: "Não" if x == 0 else "Sim",
        horizontal=True
    )

    # 4. Autônomo (radio)
    self_employed = st.radio(
        "Trabalha como autônomo(a)?",
        options=[0, 1],
        format_func=lambda x: "Não" if x == 0 else "Sim",
        horizontal=True
    )

    # 5. Educação (radio)
    education = st.radio(
        "Nível de escolaridade:",
        options=[0, 1],
        format_func=lambda x: "Graduado" if x == 0 else "Não graduado",
        horizontal=True
    )

    # 6. Rendimento (text)
    applicant_income = st.text_input(
        "Rendimento mensal (em R$):",
        placeholder="Ex.: 5000"
    )

    # 7. Valor do empréstimo (text)
    loan_amount = st.text_input(
        "Valor do empréstimo pretendido (em R$):",
        placeholder="Ex.: 150000 ou 150 (em milhares)",
        help="Valores acima de 1.000 serão convertidos automaticamente para milhares, conforme a base de treino."
    )

    enviar = st.form_submit_button(
        "Analisar Empréstimo",
        use_container_width=True
    )

# Processamento e previsão
if enviar:

    if not applicant_income.strip() or not loan_amount.strip():
        st.warning("⚠️ Por favor, informe o rendimento mensal e o valor do empréstimo.")
        st.stop()

    try:
        # Converter valores numéricos
        rendimento = float(applicant_income.replace(".", "").replace(",", "."))
        valor_emprestimo = float(loan_amount.replace(".", "").replace(",", "."))

        if rendimento <= 0 or valor_emprestimo <= 0:
            st.error("Informe valores maiores que zero para rendimento e empréstimo.")
            st.stop()

        # Ajuste de escala: no dataset de treino, LoanAmount está em milhares
        valor_emprestimo_modelo = valor_emprestimo / 1000.0 if valor_emprestimo >= 1000 else valor_emprestimo

        # Montar o DataFrame exatamente na estrutura e ordem do treinamento
        dados = {
            'Gender': gender,
            'Married': married,
            'Dependents': int(dependents),
            'Education': education,
            'Self_Employed': self_employed,
            'ApplicantIncome': rendimento,
            'LoanAmount': valor_emprestimo_modelo
        }

        novos_dados = pd.DataFrame([dados])

        # Realizar previsão
        previsao = int(modelo.predict(novos_dados)[0])

        st.divider()
        st.subheader("Resultado da Análise")

        # Status exigido no item (k): sim / não
        if previsao == 1:
            st.success("### ✅ Status: Sim (Empréstimo Aprovado)")
            st.write("O perfil atende aos critérios do modelo de liberação de crédito.")
        else:
            st.error("### ❌ Status: Não (Empréstimo Não Aprovado)")
            st.write("O perfil não atingiu os critérios mínimos para liberação no momento.")

        # Probabilidade da classe classificada (Item k)
        if hasattr(modelo, "predict_proba"):
            probabilidades = modelo.predict_proba(novos_dados)[0]
            classes = list(modelo.classes_)
            indice_pred = classes.index(previsao)
            confianca = probabilidades[indice_pred] * 100

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    label="Confiança da Decisão",
                    value=f"{confianca:.2f}%"
                )
            with col2:
                idx_aprov = classes.index(1) if 1 in classes else 0
                prob_aprov = probabilidades[idx_aprov] * 100
                st.metric(
                    label="Chance de Aprovação (Classe 1)",
                    value=f"{prob_aprov:.2f}%"
                )

            st.progress(confianca / 100.0)

        else:
            if hasattr(modelo, "decision_function"):
                score = modelo.decision_function(novos_dados)[0]
                st.info(f"Modelo linear sem probabilidades diretas. Score de decisão: `{score:.4f}`")

    except ValueError:
        st.error(
            "Verifique os campos numéricos. Informe valores válidos para rendimento e empréstimo."
        )
    except Exception as erro:
        st.error(f"Erro ao realizar análise: {erro}")