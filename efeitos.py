# -*- coding: utf-8 -*-
"""
Dashboard de Multiplicadores de Metas Diárias
Geração da tabela de multiplicadores baseada no histórico e efeitos sazonais.
"""
from __future__ import annotations

import calendar
import datetime
from typing import Any, Dict

import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Dicionário do Modelo Treinado (Preservado integralmente)
# -----------------------------------------------------------------------------
FUNIL_MODELO_PRODUCAO: Dict[str, Any] = {
    'schema_version': 'funil_elasticnet_cal_lags_v1',
    'conjunto': 'cal_lags',
    'alpha': 0.5,
    'l1_ratio': 0.3,
    'incluir_mes': True,
    'feature_names': [
        'cal_0', 'cal_1', 'cal_2', 'cal_3', 'cal_4', 'cal_5', 'cal_6', 'cal_7',
        'cal_8', 'cal_9', 'cal_10', 'cal_11', 'cal_12', 'cal_13', 'cal_14', 'cal_15',
        'cal_16', 'cal_17', 'cal_18', 'cal_19', 'cal_20', 'cal_21', 'cal_22', 'cal_23',
        'cal_24', 'cal_25', 'cal_26', 'cal_27', 'cal_28', 'cal_29', 'cal_30', 'cal_31',
        'cal_32', 'cal_33', 'cal_34', 'cal_35', 'cal_36', 'cal_37', 'cal_38', 'cal_39',
        'cal_40', 'cal_41', 'cal_42', 'cal_43', 'cal_44', 'cal_45', 'cal_46', 'cal_47',
        'cal_48', 'cal_49', 'agendamentos_lag1_7', 'agendamentos_lag8_14',
        'agendamentos_lag15_21', 'agendamentos_lag22_28', 'visitas_lag1_7',
        'visitas_lag8_14', 'visitas_lag15_21', 'visitas_lag22_28', 'pastas_lag1_7',
        'pastas_lag8_14', 'pastas_lag15_21', 'pastas_lag22_28',
        'pastas_aprovadas_lag1_7', 'pastas_aprovadas_lag8_14',
        'pastas_aprovadas_lag15_21', 'pastas_aprovadas_lag22_28', 'vendas_lag1_7',
        'vendas_lag8_14', 'vendas_lag15_21', 'vendas_lag22_28',
    ],
    'coefs': {
        'agendamentos': [
            -11.050023757122998, 1.038487609988671, -0.418124043450884, 0.442395732483458, 0.0, 1.5043060233127707, -4.79664719604406, -5.604638520549051,
            0.3307953773367116, -0.6395487540339839, 0.0, 1.0386498125055847, 1.1726599921675265, -0.0, -4.698518480368115, -1.0308131825149127,
            3.6445785363732806, -1.447882060382428, -7.204983402900458, 0.0, 3.651255151779177, -3.5088313120886867, -0.0, 8.244037224651322,
            0.0, 0.0, 3.700517849269758, 5.993521345091159, -0.34789308136715963, 7.8649989063544545, -2.91418811651284, -11.801019009446438,
            -1.4794439627409222, -10.98289209693722, 50.66223419883655, 4.591966028051389, 0.27513531844632166, -29.79641318877947, -2.032225387855265, -0.6330064208964424,
            8.419611128654562, 6.163075652020995, -0.7073162326061971, -0.0, 3.133611441766049, 0.7368229250218166, 0.6725987573442797, 1.3207690091877553,
            -3.3988118810412016, -12.80896936444073, 0.028546673248611723, 0.017664390956704096, 0.0074582695645180296, 0.011935259954292352, 0.07354415744056826, 0.0392253539742714,
            0.022737692506707458, 0.0127343370903805, 0.02552922921923293, 0.0, -0.002926447724372564, -0.0, -0.023879040237688397, 0.0,
            0.0, -0.009814685318714121, -0.13605947923178324, -0.11051715958782876, -0.0480291768980494, -0.1111889736858423, 25.518674435130997,
        ],
        'visitas': [
            -0.1446158706482313, 0.0, -0.6576659613751795, 0.1389540086584349, -0.377873533205042, -0.146361483177348, -0.9987717886271178, -1.0849597046249826,
            0.0, -0.08800284212767584, 0.0, -0.0, -0.0, -0.0, -0.6446532418450247, -0.0,
            -0.0, 1.2697311523488617, -0.0, 1.058348499769358, 0.0, -0.0, 1.373782691160814, 0.2591807150881517,
            -0.0, 0.0, 0.8108179657369299, 1.3478352049918847, -0.0, 0.5569768904217159, -0.0, -1.1072165391982218,
            -2.308244319772371, -3.960928349127075, -3.113707031167763, -1.0535131169204035, 18.6605747509046, -1.0550990313684698, 1.596962357264277, -1.298001927847425,
            0.47226600860013807, -0.0, 0.0, -2.0851054556848343, -0.584850135478679, -0.0, 0.5413827948190845, 2.0906848416485118,
            1.1774488379230574, -3.3758146603146897, 0.0074536506421286125, 0.003978534663514848, 0.001880905278990792, 0.00236741202701496, 0.030892313051985438, 0.02112699072691853,
            0.009491838610875316, 0.007083399341173249, 0.01385231404697147, -0.0024286416883260793, -0.0, 0.0009979376020420963, 0.0008007206085815955, -0.0,
            0.0, 0.0, -0.03218068527766599, -0.017397647848058598, -0.021067391256373167, -0.029106308549339335, 6.760799775315082,
        ],
        'pastas': [
            -2.257660160372139, 0.37804668806549585, -0.8188908909767566, -0.7911168539879757, -0.0, -0.6276025072557446, -0.5693125888781922, -0.0,
            0.36730776869997234, -0.6090992493485636, 0.0, -0.0, 0.0, -0.6407029879254541, -0.22363716339833273, -0.9959623769656082,
            -0.0995014340792171, 0.0, -0.4446914006054063, 0.053702775562173344, 0.0, -0.0, 0.25681307624841726, 0.4879161608206745,
            0.25974634112636374, 2.829001296348825, 3.3903746625625044, 1.6216671625960872, 0.0, 1.5054829998167254, 0.0, 0.0,
            2.169910744495752, 0.8866311180081654, 2.5190340402856277, 2.568863992477152, 1.8252468917425655, -15.952118304331036, 4.089806676160004, -0.0,
            3.0244750667088462, 0.0, 0.0, -0.21947233759338208, -1.4048245833108872, 0.0, -1.145801155726143, -0.9141429197182945,
            0.5667724453793453, -3.735056172948177, 0.0011883536587739955, -0.0, -0.0, -0.0, 0.0038239110262645822, 0.0,
            -0.0, -0.0, 0.053076383684951864, 0.01628370008719105, 0.010181029816810011, 0.01134665219209089, 0.01919491590431224, 0.0,
            0.0034907449374374897, -0.003983682455728812, -0.0, -0.0, 0.0, 0.0, 8.541321401889125,
        ],
        'pastas_aprovadas': [
            -0.0, 1.286418801392201, -0.0, 0.1096047048215759, -0.0, -0.0, -0.0, -0.0,
            -0.0, -0.6873312685412176, -0.0, 0.0, -0.0, -0.45349523807398806, 0.0, -0.0,
            0.0, -0.22197244123571663, -0.6668303696314954, 0.0, 0.0, -0.023492855597622053, 0.41712895519882753, -0.173797479705193,
            -0.0, 0.9175906237994016, 0.42912541252567155, 0.5037806961811774, 0.0, 0.04251140271987944, 0.0, 1.2070697707842875,
            -0.0, 0.004452492087191327, 0.0, 0.9641332371870946, 0.28237430750078957, -6.032825556173662, 0.6036149400664597, -0.2409258248307588,
            0.0, 0.4605254209469333, 0.49979582870123335, 0.0, -0.0, 0.0, 0.0, -0.18869037029264502,
            -0.2726160873448889, -0.6656027990114094, 0.00013888108779133894, 0.0, 0.0, 0.0, 0.0012466375312928557, 0.00042487296661787815,
            -0.0, -0.0, 0.0211420449696927, 0.00404804179782973, 0.0, 0.00029221606057710744, 0.019431117704225863, 0.012134450327900354,
            0.0, 0.0, -0.0, 0.0, 0.0, 0.005232421713483153, 2.8626292290311195,
        ],
        'vendas': [
            0.20528593276015503, 0.698417783067199, 0.3675451925045679, 0.321792704983368, -0.0, -0.0, -0.0, 0.0,
            -0.0, -0.8532504518005523, -0.053521301656589367, -0.0, 0.0, -0.20691941235240396, -0.0, -0.0,
            -0.0, -0.0, -0.13175897883196047, 0.0, -0.0, -0.0, -0.0, 0.0,
            0.18071359527530556, 0.0, 0.0, 0.1894548720542877, 0.0, 0.389496984057287, -0.0, 0.35908831598201363,
            -0.0, -0.061743672343900026, -0.09141691518218586, 0.0, 2.0783056039277445, -1.8354991984619378, 0.0, -0.12725787236565514,
            -0.0, 0.467109201218153, 0.33104757574626836, 0.0, -0.0, -0.003614909853667342, -0.0, -0.0,
            0.0009870482915211455, -0.0, -0.0, -0.00021316560066822872, -0.0005062140142649687, -0.00027505732719980804, -0.0, -0.0,
            -0.0015675284840701315, -0.0009382118647754134, 0.012169596966085192, 0.0, -0.0, 0.0, 0.008202816037259452, -0.0,
            -0.0, 0.0, 0.014770382036083303, 0.011190452780320471, 0.003443453390416612, 0.01630494307484225, 1.5042820591847148,
        ],
    },
    'totais_hist': {
        'agendamentos': 81282.40120165226,
        'visitas': 30094.151408186248,
        'pastas': 30925.2614344724,
        'pastas_aprovadas': 11212.096582801352,
        'vendas': 5826.615170859932,
    },
}

# -----------------------------------------------------------------------------
# Dicionários Auxiliares de Datas
# -----------------------------------------------------------------------------
DIAS_SEMANA_PT = {
    0: "Segunda-feira",
    1: "Terça-feira",
    2: "Quarta-feira",
    3: "Quinta-feira",
    4: "Sexta-feira",
    5: "Sábado",
    6: "Domingo"
}

MESES_PT = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
    5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
    9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

# -----------------------------------------------------------------------------
# Lógica de Cálculo dos Multiplicadores
# -----------------------------------------------------------------------------
def get_beta_dia(indicador: str, dia_mes: int, dia_semana: int, mes: int) -> float:
    """
    Obtém o beta bruto de um dia específico considerando os coeficientes do modelo.
    Soma do intercepto + efeito do dia do mês + efeito do dia da semana + efeito do mês.
    """
    coefs = FUNIL_MODELO_PRODUCAO['coefs'][indicador]
    
    intercepto = coefs[70]
    beta_dia = coefs[dia_mes - 1]          # Índices de 0 a 30 (Dias 1 a 31)
    beta_dow = coefs[31 + dia_semana]      # Índices de 31 a 37 (Seg a Dom)
    beta_mes = coefs[38 + mes - 1]         # Índices de 38 a 49 (Jan a Dez)
    
    # Garantir que o peso base seja no mínimo um pequeno valor positivo para evitar divisão por zero
    return max(0.0001, intercepto + beta_dia + beta_dow + beta_mes)

def calcular_tabela_multiplicadores() -> pd.DataFrame:
    hoje = datetime.date.today()
    ano_atual = hoje.year
    mes_atual = hoje.month
    
    dias_no_mes = calendar.monthrange(ano_atual, mes_atual)[1]
    indicadores = ["agendamentos", "visitas", "pastas", "pastas_aprovadas"]
    
    # 1. Calcular os betas brutos e a soma para cada indicador no mês atual
    betas_brutos = {ind: {} for ind in indicadores}
    soma_betas = {ind: 0.0 for ind in indicadores}
    
    for d in range(1, dias_no_mes + 1):
        data_loop = datetime.date(ano_atual, mes_atual, d)
        weekday = data_loop.weekday()
        
        for ind in indicadores:
            beta = get_beta_dia(ind, d, weekday, mes_atual)
            betas_brutos[ind][d] = beta
            soma_betas[ind] += beta
            
    # 2. Calcular a conversão histórica inversa (Indicador / Vendas) para os últimos 36 meses
    vendas_hist = FUNIL_MODELO_PRODUCAO['totais_hist']['vendas']
    inv_conv = {}
    for ind in indicadores:
        total_ind = FUNIL_MODELO_PRODUCAO['totais_hist'][ind]
        inv_conv[ind] = total_ind / vendas_hist if vendas_hist > 0 else 0.0
        
    # 3. Construir a tabela de resultados com os multiplicadores finais
    resultados = []
    
    for d in range(1, dias_no_mes + 1):
        data_loop = datetime.date(ano_atual, mes_atual, d)
        weekday = data_loop.weekday()
        
        linha = {
            "Mês": MESES_PT[mes_atual],
            "Dia do Mês": d,
            "Dia da Semana": DIAS_SEMANA_PT[weekday],
        }
        
        for ind in indicadores:
            peso_diario = betas_brutos[ind][d] / soma_betas[ind] if soma_betas[ind] > 0 else 0.0
            multiplicador = inv_conv[ind] * peso_diario
            
            nome_col = f"Multiplicador {ind.replace('_', ' ').title()}"
            linha[nome_col] = multiplicador
            
        resultados.append(linha)
        
    return pd.DataFrame(resultados)

# -----------------------------------------------------------------------------
# Interface Streamlit
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Multiplicadores de Metas do Funil",
        page_icon="📊",
        layout="wide"
    )
    
    COR_AZUL_ESC = "#04428f"
    COR_VERMELHO = "#e30613"

    st.markdown(
        f"""
        <h2 style='text-align: center; color: {COR_AZUL_ESC};'>
            Multiplicadores Diários do Funil Comercial
        </h2>
        """, 
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        Esta tabela apresenta o multiplicador da meta para cada dia do mês corrente. 
        A lógica utiliza as conversões históricas e pondera a meta diária utilizando 
        os efeitos de mês, dia do mês e dia da semana extraídos do modelo regressivo.
        
        **Cálculo:** `Multiplicador Final = (1 / Conversão Histórica) * Peso Diário`
        
        *(Para obter a meta diária do indicador, multiplique a Meta de Vendas Total do Mês pelo multiplicador correspondente no dia desejado).*
        """
    )
    
    with st.spinner("Calculando a matriz de multiplicadores sazonais..."):
        df_tabela = calcular_tabela_multiplicadores()
        
    st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:1rem 0;'/>", unsafe_allow_html=True)
    
    st.dataframe(
        df_tabela,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Mês": st.column_config.TextColumn("Mês"),
            "Dia do Mês": st.column_config.NumberColumn("Dia do Mês", format="%d"),
            "Dia da Semana": st.column_config.TextColumn("Dia da Semana"),
            "Multiplicador Agendamentos": st.column_config.NumberColumn("Multiplicador Agendamentos", format="%.5f"),
            "Multiplicador Visitas": st.column_config.NumberColumn("Multiplicador Visitas", format="%.5f"),
            "Multiplicador Pastas": st.column_config.NumberColumn("Multiplicador Pastas", format="%.5f"),
            "Multiplicador Pastas Aprovadas": st.column_config.NumberColumn("Multiplicador Pastas Aprovadas", format="%.5f"),
        }
    )

if __name__ == "__main__":
    main()
