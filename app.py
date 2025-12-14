import streamlit as st
import pandas as pd
import streamlit as st
import pandas as pd
import altair as alt
from openai import OpenAI
import google.generativeai as genai
import anthropic
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import re
import base64
from typing import Dict, List, Tuple, Optional
from investment_map import create_investment_map

st.set_page_config(
    page_title="TRIDENT AI: Gayrimenkul Zekası",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AIClients:
    """Üç farklı AI gibi görünen ama hepsi OpenAI olan sınıf"""
    
    def __init__(self):
        self.clients = {}
        self.roles = {
            'openai': {
                'name': 'TRIDENT Finans Analisti',
                'emoji': '💰',
                'description': 'Fiyat analizi ve yatırım tavsiyeleri',
                'personality': 'Finans odaklı, rakamlarla konuşan, yatırım perspektifli'
            },
            'anthropic': {
                'name': 'TRIDENT Emlak Danışmanı',
                'emoji': '🏠',
                'description': 'Yaşam kalitesi ve semt analizleri',
                'personality': 'Sıcak, samimi, yaşam kalitesine odaklı, detaycı'
            },
            'google': {
                'name': 'TRIDENT Teknik Uzman',
                'emoji': '🔧',
                'description': 'Teknik detaylar ve risk analizleri',
                'personality': 'Teknik, analitik, veri odaklı, mühendis bakış açılı'
            }
        }
        

        self._initialize_clients()
    
    def _initialize_clients(self):
        """Tüm AI client'larını başlat - HEPSİ OPENAI"""
        if "OPENAI_API_KEY" in st.secrets:
            try:
           
                main_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
                
                self.clients['openai'] = main_client
                self.clients['anthropic'] = main_client  
                self.clients['google'] = main_client     
                
                st.session_state.openai_available = True
                st.session_state.anthropic_available = True  
                st.session_state.google_available = True     
                
            except Exception as e:
                st.error(f"⚠️ OpenAI API Hatası: {e}")
                st.session_state.openai_available = False
                st.session_state.anthropic_available = False
                st.session_state.google_available = False
        else:
            st.error("⚠️ HATA: secrets.toml dosyasında 'OPENAI_API_KEY' bulunamadı.")
            st.session_state.openai_available = False
            st.session_state.anthropic_available = False
            st.session_state.google_available = False
        
        if not st.session_state.get('openai_available', False):
            st.error("❌ OPENAI ÇALIŞMIYOR! Lütfen API anahtarını kontrol edin.")
            st.stop()

ai_clients = AIClients()

def get_openai_response_with_personality(prompt: str, personality: str, context: str = "", 
                                        use_vision: bool = False, image_data: Optional[str] = None) -> str:
    """
    OpenAI'yi farklı kişiliklerde kullan
    """
    try:
        system_message = f"""
        Sen bir gayrimenkul uzmanısın ama şu özelliklere sahipsin:
        {personality}
        
        Cevabını bu kişiliğe uygun ver.
        Kullanıcı sorununu/analizini bu bakış açısıyla değerlendir.
        """
        
        full_prompt = f"{context}\n\nKullanıcı: {prompt}"
        
        if use_vision and image_data:
            response = ai_clients.clients['openai'].chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": full_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.7
            )
        else:
            
            response = ai_clients.clients['openai'].chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": full_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"⚠️ AI Hatası: {str(e)[:100]}"

def get_ai_responses(prompt: str, context: str = "", use_vision: bool = False, image_data: Optional[str] = None) -> Dict[str, str]:
    """
    Üç farklı AI gibi görünen ama hepsi OpenAI olan yanıtlar
    """
    responses = {}
    
    if st.session_state.get('openai_available', False):
        try:
            openai_context = f"""
            {ai_clients.roles['openai']['description']} rolündesin. 
            {ai_clients.roles['openai']['name']} olarak cevap ver.
            
            {context}
            """
            
            responses['openai'] = get_openai_response_with_personality(
                prompt=prompt,
                personality=ai_clients.roles['openai']['personality'],
                context=openai_context,
                use_vision=use_vision,
                image_data=image_data
            )
        except Exception as e:
            responses['openai'] = f"💰 Finans Analisti: {str(e)[:100]}"
    
    if st.session_state.get('anthropic_available', False):
        try:
            anthropic_context = f"""
            {ai_clients.roles['anthropic']['description']} rolündesin. 
            {ai_clients.roles['anthropic']['name']} olarak cevap ver.
            
            {context}
            
            NOT: Cevabını {ai_clients.roles['anthropic']['emoji']} ile başlat.
            """
            
            responses['anthropic'] = get_openai_response_with_personality(
                prompt=prompt,
                personality=ai_clients.roles['anthropic']['personality'],
                context=anthropic_context,
                use_vision=False,  # Anthropic vision yok
                image_data=None
            )
        except Exception as e:
            responses['anthropic'] = f"🏠 Emlak Danışmanı: {str(e)[:100]}"
    
    # Google (Teknik Uzman) - ASLINDA OPENAI
    if st.session_state.get('google_available', False):
        try:
            google_context = f"""
            {ai_clients.roles['google']['description']} rolündesin. 
            {ai_clients.roles['google']['name']} olarak cevap ver.
            
            {context}
            
            NOT: Cevabını {ai_clients.roles['google']['emoji']} ile başlat.
            Rakamlarla konuş, teknik detaylara odaklan.
            """
            
            responses['google'] = get_openai_response_with_personality(
                prompt=prompt,
                personality=ai_clients.roles['google']['personality'],
                context=google_context,
                use_vision=False,  # Google vision yok
                image_data=None
            )
        except Exception as e:
            responses['google'] = f"🔧 Teknik Uzman: {str(e)[:100]}"
    
    return responses

def get_ai_report_advanced(district: str, room: str, m2: float, pred: float, actual: float, advice: str) -> Dict[str, str]:
    """
    Geliştirilmiş AI raporu - 3 farklı AI'dan
    """
    context = f"""
    Emlak Analizi Verileri:
    - 📍 Konum: {district}
    - 🏠 Oda Tipi: {room}
    - 📐 Metrekare: {m2}m²
    - 💰 TRIDENT Tahmini: {pred:,.0f} TL
    - 🏷️ İlan Fiyatı: {actual:,.0f} TL
    - ⚖️ Durum: {advice}
    
    Fiyat Farkı: {((actual/pred)-1)*100:.1f}%
    """
    
    prompt = f"Bu gayrimenkul hakkında detaylı analiz yap ve yatırım tavsiyesi ver."
    
    return get_ai_responses(prompt, context)

def get_comparison_ai_analysis(option_a: dict, option_b: dict, priority: str) -> Dict[str, str]:
    """
    Karşılaştırma için AI analizi
    """
    context = f"""
    İKİ GAYRİMENKUL KARŞILAŞTIRMASI:
    
    🅰️ SEÇENEK A:
    - İlçe: {option_a['district']}
    - Mahalle: {option_a['neighborhood']}
    - Oda: {option_a['room']}
    - m²: {option_a['m2']}
    - Fiyat: {option_a['price']:,.0f} TL
    - TRIDENT Adil Değer: {option_a['pred']:,.0f} TL
    """
    
    if 'security_info' in option_a and option_a['security_info']:
        context += f"""
        - 🛡️ Güvenlik Risk: {option_a['security_info'].get('risk_seviyesi', 'Bilinmiyor')}
        - 📊 Suç Sayısı: {option_a['security_info'].get('suc_sayisi', 'Bilinmiyor')}
        """
    
    context += f"""
    
    🅱️ SEÇENEK B:
    - İlçe: {option_b['district']}
    - Mahalle: {option_b['neighborhood']}
    - Oda: {option_b['room']}
    - m²: {option_b['m2']}
    - Fiyat: {option_b['price']:,.0f} TL
    - TRIDENT Adil Değer: {option_b['pred']:,.0f} TL
    """
    
    if 'security_info' in option_b and option_b['security_info']:
        context += f"""
        - 🛡️ Güvenlik Risk: {option_b['security_info'].get('risk_seviyesi', 'Bilinmiyor')}
        - 📊 Suç Sayısı: {option_b['security_info'].get('suc_sayisi', 'Bilinmiyor')}
        """
    
    context += f"""
    
    🎯 Kullanıcı Önceliği: {priority}
    """
    
    prompt = "Bu iki seçeneği karşılaştır ve kullanıcının önceliğine göre hangisinin daha iyi olduğunu açıkla. Güvenlik risklerini de değerlendir."
    
    return get_ai_responses(prompt, context)

def get_personal_assistant_ai(budget: float, work_location: str, family_type: str, 
                              transport: str, lifestyle: List[str], amenities: List[str], 
                              affordable_districts: List[str]) -> Dict[str, str]:
    """
    Kişisel asistan için AI analizi
    """
    context = f"""
    KULLANICI PROFİLİ:
    - 💰 Bütçe: {budget:,.0f} TL
    - 📍 İş/Konum: {work_location}
    - 👨‍👩‍👧‍👦 Aile Tipi: {family_type}
    - 🚇 Ulaşım: {transport}
    - 🎭 Yaşam Tarzı: {', '.join(lifestyle)}
    - 🏥 Önemli Olanaklar: {', '.join(amenities)}
    
    BÜTÇEYE UYGUN İLÇELER: {', '.join(affordable_districts[:10])}
    """
    
    prompt = "Bu kullanıcı için İstanbul'da en uygun 3 bölge öner ve detaylı açıkla."
    
    return get_ai_responses(prompt, context)

def get_disaster_risk_ai(district: str, neighborhood: str, building_age: str, 
                         floor_location: str) -> Dict[str, str]:
    """
    Afet risk analizi için AI
    """
    context = f"""
    BİNA BİLGİLERİ:
    - 📍 İlçe: {district}
    - 🏘️ Mahalle: {neighborhood}
    - 🏗️ Bina Yaşı: {building_age}
    - 🏢 Kat: {floor_location}
    """
    
    prompt = "Bu bina için deprem risk analizi yap, zemin etüdü tavsiyeleri ver ve risk skoru oluştur."
    
    return get_ai_responses(prompt, context)

@st.cache_data
def load_and_clean_data():
    try:
        df = pd.read_csv("hackathon_train_set.csv", sep=";")
        
        if 'Available for Loan' in df.columns:
            df = df[df['Available for Loan'] == 'Yes']
        
        df['Price_Clean'] = df['Price'].astype(str).str.replace('.', '', regex=False).str.replace(' TL', '', regex=False).str.strip()
        df['Price_Clean'] = pd.to_numeric(df['Price_Clean'], errors='coerce')
        df['m² (Net)'] = pd.to_numeric(df['m² (Net)'], errors='coerce')
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].fillna(df[col].median())
            
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'Price':
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Bilinmiyor')

        def get_room_sort_value(text):
            text = str(text).strip()
            if "Studio" in text or "1+0" in text: return 0.9
            nums = re.findall(r'\d+', text)
            if len(nums) >= 2:
                return float(nums[0]) + (float(nums[1]) * 0.1)
            elif len(nums) == 1:
                return float(nums[0])
            return 0

        df['Room_Sort_Value'] = df['Number of rooms'].apply(get_room_sort_value)
        
        df = df.dropna(subset=['Price_Clean', 'm² (Net)', 'Room_Sort_Value', 'District', 'Neighborhood'])
        df = df[df['Price_Clean'] > 10000]
        
        return df
    except Exception as e:
        st.error(f"Veri Yükleme Hatası: {e}")
        return pd.DataFrame()

@st.cache_data
def load_security_data():
    try:
        security_df1 = pd.read_csv("sucre_gore_ilceler.csv", encoding='utf-8')
        
        security_df2 = pd.read_csv("karakol_bazli_suclar.csv", encoding='utf-8')
        
        security_df = pd.merge(security_df1, security_df2[['ilce', 'suc_sayisi_2025_ilk9ay']], 
                              on='ilce', how='left')
        
        return security_df
    except Exception as e:
        st.warning(f"Güvenlik verileri yüklenemedi: {e}")
        return pd.DataFrame()

df = load_and_clean_data()
if df.empty: st.stop()

security_df = load_security_data()

@st.cache_resource
def train_model(data):
    le_district = LabelEncoder()
    le_neighborhood = LabelEncoder()
    
    data['District_Code'] = le_district.fit_transform(data['District'])
    data['Neighborhood_Code'] = le_neighborhood.fit_transform(data['Neighborhood'])
    
    X = data[['District_Code', 'Neighborhood_Code', 'Room_Sort_Value', 'm² (Net)']]
    y = data['Price_Clean']
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    )
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    return model, le_district, le_neighborhood, r2, rmse

model, le_dist, le_neigh, model_r2, model_rmse = train_model(df.copy())

def get_investment_advice(predicted, actual, dataset=None, trained_model=None):
    if actual <= 0: return "Veri Bekleniyor", "gray"
    
    threshold = 0.15
    if dataset is not None and trained_model is not None:
        try:
            residuals_std = model_rmse / predicted
            threshold = 0.10 + (residuals_std * 0.5)
            threshold = min(max(threshold, 0.10), 0.25)
        except:
            pass

    ratio = actual / predicted
    
    if ratio < (1 - threshold): return "FIRSAT (Opportunity) 🌟", "green"
    elif ratio > (1 + threshold): return "PAHALI (Overpriced) 🔴", "red"
    else: return "NORMAL (Fair Value) 🔵", "blue"

st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🦅 TRIDENT AI MULTI-PERSONALITY</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>3 Farklı Uzmanlıkta Yapay Zeka ile Gayrimenkul İstihbaratı</p>", unsafe_allow_html=True)

with st.sidebar:
    st.header("🎛️ Parametre Paneli")
    st.info("Her AI farklı bir uzmanlık alanından analiz yapar!")
    
    st.markdown("---")
    st.header("🤖 AI Uzmanları")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("💰 Finans", "✅")
        st.caption("Fiyat Analizi")
    with col2:
        st.metric("🏠 Emlak", "✅")
        st.caption("Yaşam Kalitesi")
    with col3:
        st.metric("🔧 Teknik", "✅")
        st.caption("Risk Analizi")
    
    st.markdown("---")
    st.header("📊 Model Performansı")
    st.metric("R² Başarısı", f"%{model_r2*100:.1f}")
    st.metric("Hata Payı (RMSE)", f"±{model_rmse:,.0f} TL")
    st.caption(f"Eğitilen Veri: {len(df):,} kayıt")

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🚀 YATIRIM DANIŞMANI", 
    "📊 GELİŞMİŞ EDA", 
    "🎯 KİŞİSEL ASİSTAN", 
    "⚖️ KARŞILAŞTIRMA", 
    "🌪️ RİSK ANALİZİ", 
    "💬 VERİ SOHBETİ",
    "📸 FOTOĞRAF ANALİZİ",
    "🗺️ YATIRIM HARİTASI"  
])

with tab1:
    col_param, col_result = st.columns([1, 2])
    with col_param:
        st.subheader("Girdi Paneli")
        input_district = st.selectbox("📍 İlçe", le_dist.classes_)
        valid_neighs = sorted(df[df['District'] == input_district]['Neighborhood'].unique())
        input_neigh = st.selectbox("🏘️ Mahalle", valid_neighs, index=0)
        sorted_rooms = df[['Number of rooms', 'Room_Sort_Value']].drop_duplicates().sort_values('Room_Sort_Value')
        room_options = sorted_rooms['Number of rooms'].tolist()
        def_idx = room_options.index("3+1") if "3+1" in room_options else 0
        input_room = st.selectbox("🏠 Oda Tipi", room_options, index=def_idx)
        input_m2 = st.selectbox("📐 m² (Net)", [50, 60, 75, 85, 90, 100, 110, 120, 135, 150, 180, 200, 250], index=5)
        st.markdown("---")
        input_price = st.number_input("İlan Fiyatı (TL)", value=500000, step=10000)
        btn_predict = st.button("3 UZMAN İLE ANALİZ ET", type="primary")
        
    with col_result:
        if btn_predict:
            try:
                room_data = df[df['Number of rooms'] == input_room]
                room_val = room_data['Room_Sort_Value'].iloc[0] if not room_data.empty else 3.0
                d_code = le_dist.transform([input_district])[0]
                n_code = le_neigh.transform([input_neigh])[0]
                pred_price = model.predict([[d_code, n_code, room_val, input_m2]])[0]
                advice, color = get_investment_advice(pred_price, input_price, df, model)
                
                st.success("✅ 3 Uzman ile Analiz Başladı")
                c1, c2, c3 = st.columns(3)
                c1.metric("TRIDENT Adil Değer", f"{pred_price:,.0f} TL")
                c2.metric("İlan Fiyatı", f"{input_price:,.0f} TL")
                c3.markdown(f"<h3 style='color:{color}; text-align:center;'>{advice}</h3>", unsafe_allow_html=True)
                
                with st.spinner("3 uzman analiz ediyor..."):
                    ai_responses = get_ai_report_advanced(
                        input_district, input_room, input_m2, 
                        pred_price, input_price, advice
                    )
                
                st.markdown("---")
                st.subheader("🤖 Çoklu Uzman Analizi")
                
                tabs = st.tabs([f"{ai_clients.roles['openai']['emoji']} {ai_clients.roles['openai']['name']}",
                               f"{ai_clients.roles['anthropic']['emoji']} {ai_clients.roles['anthropic']['name']}",
                               f"{ai_clients.roles['google']['emoji']} {ai_clients.roles['google']['name']}"])
                
                for i, (provider, response) in enumerate(ai_responses.items()):
                    with tabs[i]:
                        st.markdown(f"**{ai_clients.roles[provider]['description']}**")
                        st.markdown(response)
                        
            except Exception as e:
                st.error(f"Hata: {e}")

        neigh_data = df[(df['District'] == input_district) & (df['Neighborhood'] == input_neigh)]
        if neigh_data.empty: neigh_data = df[df['District'] == input_district]
        
        st.subheader(f"📊 {input_district} / {input_neigh} Analizi")
        if not neigh_data.empty:
            chart_data = neigh_data.groupby('Number of rooms')['Price_Clean'].mean().reset_index()
            chart = alt.Chart(chart_data).mark_bar().encode(
                x='Number of rooms', y='Price_Clean',
                color=alt.condition(alt.datum['Number of rooms'] == input_room, alt.value('#FF4B4B'), alt.value('#2E86C1'))
            ).properties(height=250)
            st.altair_chart(chart, use_container_width=True)

with tab2:
    st.header("📈 Veri Analizi (EDA)")
    
    c1, c2 = st.columns(2)
    
    with c1:
        dist_price = df.groupby('District')['Price_Clean'].mean().reset_index().sort_values('Price_Clean', ascending=False).head(10)
        chart1 = alt.Chart(dist_price).mark_bar().encode(
            x=alt.X('District:N', sort='-y', title='İlçe'),
            y=alt.Y('Price_Clean:Q', title='Ortalama Fiyat (TL)', axis=alt.Axis(format=',.0f')),
            color=alt.Color('Price_Clean:Q', scale=alt.Scale(scheme='blues'), legend=None)
        ).properties(
            title='En Pahalı 10 İlçe (Ortalama Fiyat)',
            height=350
        ).configure_axis(
            labelAngle=-45
        )
        st.altair_chart(chart1, use_container_width=True)
    
    with c2:
        sample_df = df.sample(min(1000, len(df)))
        chart2 = alt.Chart(sample_df).mark_circle(size=60).encode(
            x=alt.X('m² (Net):Q', title='Metrekare (Net)'),
            y=alt.Y('Price_Clean:Q', title='Fiyat (TL)', axis=alt.Axis(format=',.0f')),
            color=alt.Color('District:N', title='İlçe'),
            tooltip=['District', 'Neighborhood', 'Number of rooms', 'Price_Clean', 'm² (Net)']
        ).properties(
            title='Metrekare vs Fiyat Dağılımı',
            height=350
        ).interactive()
        st.altair_chart(chart2, use_container_width=True)
    
    st.markdown("---")
    
    c3, c4 = st.columns(2)
    
    with c3:
        room_price = df.groupby('Number of rooms')['Price_Clean'].mean().reset_index().sort_values('Price_Clean', ascending=False).head(15)
        chart3 = alt.Chart(room_price).mark_bar().encode(
            x=alt.X('Number of rooms:N', title='Oda Tipi', sort='-y'),
            y=alt.Y('Price_Clean:Q', title='Ortalama Fiyat (TL)', axis=alt.Axis(format=',.0f')),
            color=alt.Color('Price_Clean:Q', scale=alt.Scale(scheme='greens'), legend=None)
        ).properties(
            title='Oda Tipine Göre Ortalama Fiyat',
            height=300
        ).configure_axis(
            labelAngle=-45
        )
        st.altair_chart(chart3, use_container_width=True)
    
    with c4:
        neigh_counts = df['Neighborhood'].value_counts().reset_index().head(15)
        neigh_counts.columns = ['Neighborhood', 'Count']
        chart4 = alt.Chart(neigh_counts).mark_bar().encode(
            x=alt.X('Neighborhood:N', title='Mahalle', sort='-y'),
            y=alt.Y('Count:Q', title='İlan Sayısı'),
            color=alt.Color('Count:Q', scale=alt.Scale(scheme='reds'), legend=None)
        ).properties(
            title='En Çok İlan Bulunan 15 Mahalle',
            height=300
        ).configure_axis(
            labelAngle=-45
        )
        st.altair_chart(chart4, use_container_width=True)
    
    st.markdown("---")
    
    if not security_df.empty:
        st.subheader("🛡️ Güvenlik ve Risk Analizi")
        
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            risk_counts = security_df['risk_seviyesi'].value_counts().reset_index()
            risk_counts.columns = ['Risk Seviyesi', 'İlçe Sayısı']
            
            risk_chart = alt.Chart(risk_counts).mark_bar().encode(
                x='Risk Seviyesi:N',
                y='İlçe Sayısı:Q',
                color='Risk Seviyesi:N'
            ).properties(title='Risk Seviyesi Dağılımı', height=300)
            st.altair_chart(risk_chart, use_container_width=True)
        
        with col_s2:
            yaka_chart = alt.Chart(security_df).mark_bar().encode(
                x='yakasi:N',
                y='mean(suc_sayisi):Q',
                color='yakasi:N'
            ).properties(title='Yakasına Göre Ortalama Suç Sayısı', height=300)
            st.altair_chart(yaka_chart, use_container_width=True)
    
    st.subheader("📊 İstatistiksel Özet")
    
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Toplam İlan", f"{len(df):,}")
    with col_stat2:
        st.metric("Ortalama Fiyat", f"{df['Price_Clean'].mean():,.0f} TL")
    with col_stat3:
        st.metric("Ortalama m²", f"{df['m² (Net)'].mean():.1f}")
    with col_stat4:
        st.metric("Fiyat Std", f"{df['Price_Clean'].std():,.0f} TL")
    
    st.subheader("🔥 Korelasyon Matrisi")
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        corr_df = corr_matrix.reset_index().melt('index')
        corr_df.columns = ['Variable1', 'Variable2', 'Correlation']
        
        heatmap = alt.Chart(corr_df).mark_rect().encode(
            x='Variable1:N',
            y='Variable2:N',
            color=alt.Color('Correlation:Q', scale=alt.Scale(scheme='redblue', domainMid=0)),
            tooltip=['Variable1', 'Variable2', 'Correlation']
        ).properties(
            width=600,
            height=500
        )
        st.altair_chart(heatmap, use_container_width=True)

with tab3:
    st.header("🎯 'Bana Uygun Ev Nerede?' - 3 Uzman ile Akıllı Analiz")
    
    c_in, c_out = st.columns([1, 1])
    
    with c_in:
        st.markdown("### 📋 Profilinizi Oluşturun")
        with st.form("personal_search"):
            u_budget = st.number_input("Maksimum Bütçeniz (TL)", min_value=100000, max_value=50000000, value=2000000, step=50000)
            u_work = st.text_input("İş veya Okul Konumunuz", placeholder="Örn: Maslak, Levent, Kadıköy")
            
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                u_family = st.selectbox("Hane Tipi", ["Tek Kişi (Öğrenci/Çalışan)", "Çift", "Çekirdek Aile (Çocuklu)", "Geniş Aile", "Ev Arkadaşları"])
            with col_f2:
                u_transport = st.selectbox("Ulaşım Tercihi", ["Metro/Toplu Taşıma Şart", "Özel Araç Kullanıyorum", "Yürüme Mesafesi", "Farketmez"])
            
            u_style = st.multiselect("Yaşam Tarzı Beklentileri", 
                                     ["Sessiz & Sakin", "Gece Hayatı & Eğlence", "Deniz Manzarası / Sahil", "Doğa & Yeşil Alan", 
                                      "Site İçi & Güvenlik", "Lüks & Konfor", "Öğrenci Dostu / Ekonomik"],
                                     default=["Sessiz & Sakin"])
            
            u_amenities = st.multiselect("Olmazsa Olmaz Yakınlıklar", 
                                         ["Metro İstasyonu", "AVM", "Hastane", "Okul/Kreş", "Spor Salonu", "Park", "Market/Pazar"],
                                         default=["Metro İstasyonu"])
            
            submitted = st.form_submit_button("🔍 3 UZMAN İLE ANALİZ ET", type="primary")
            
    with c_out:
        st.markdown("### 🦅 3 Uzman Analiz Sonucu")
        if submitted:
            affordable_df = df[df['Price_Clean'] <= u_budget]
            aff_districts = affordable_df['District'].unique().tolist()
            
            if not aff_districts:
                st.error("😔 Belirttiğiniz bütçeye uygun veri setimizde hiç ilan bulunamadı.")
            else:
                with st.spinner("3 uzman (Finans, Emlak, Güvenlik) verileri analiz ediyor..."):
                    try:
                        
                        district_details = []
                        
                        for dist in aff_districts[:15]:
                            dist_data = affordable_df[affordable_df['District'] == dist]
                            avg_price = dist_data['Price_Clean'].mean()
                            count_ads = len(dist_data)
                            
                            risk_txt = "Veri Yok"
                            suc_sayisi = "Bilinmiyor"
                            if not security_df.empty:
                                sec_row = security_df[security_df['ilce'] == dist]
                                if not sec_row.empty:
                                    risk_txt = sec_row.iloc[0]['risk_seviyesi']
                                    suc_sayisi = sec_row.iloc[0]['suc_sayisi']
                            
                            district_details.append(
                                f"- {dist}: Ort. Fiyat {avg_price:,.0f} TL ({count_ads} ilan) | Güvenlik Riski: {risk_txt} (Suç: {suc_sayisi})"
                            )
                        
                        formatted_districts = "\n".join(district_details)
                        
                        enhanced_context = f"""
                        KULLANICI PROFİLİ:
                        - 💰 Bütçe: {u_budget:,.0f} TL
                        - 📍 İş/Konum: {u_work}
                        - 👨‍👩‍👧‍👦 Aile Tipi: {u_family}
                        - 🚇 Ulaşım: {u_transport}
                        - 🎭 Yaşam Tarzı: {', '.join(u_style)}
                        - 🏥 Beklentiler: {', '.join(u_amenities)}
                        
                        VERİ TABANIMIZDAKİ ADAY İLÇELERİN GERÇEK DURUMU (Python Analizi):
                        {formatted_districts}
                        
                        GÖREV:
                        Yukarıdaki 'GERÇEK DURUM' verilerini kullanarak bu kullanıcıya en uygun 3 ilçeyi seç.
                        - Finans Uzmanı olarak: Fiyat/Performans dengesini gözet.
                        - Güvenlik Uzmanı olarak: Eğer aile ise 'Yüksek Risk'li yerleri ele.
                        - Emlak Uzmanı olarak: Yaşam tarzına uygunluğu değerlendir.
                        """
                        
                        ai_responses = get_ai_responses(
                            prompt="Verilen istatistiklere dayanarak bana en uygun 3 bölgeyi detaylı öner.", 
                            context=enhanced_context
                        )
                        
                        st.success("✅ 3 Uzman Analizleri Geldi!")
                        
                        tabs = st.tabs([f"{ai_clients.roles['openai']['emoji']} Finansal",
                                       f"{ai_clients.roles['anthropic']['emoji']} Yaşamsal",
                                       f"{ai_clients.roles['google']['emoji']} Teknik"])
                        
                        for i, (provider, response) in enumerate(ai_responses.items()):
                            with tabs[i]:
                                st.markdown(f"**{ai_clients.roles[provider]['name']}**")
                                st.markdown(response)
                        
                        st.divider()
                        
                        if not security_df.empty:
                            st.subheader("🛡️ Güvenlik Risk Analizi")
                            
                            risky_districts = []
                            for district in aff_districts[:10]:
                                district_risk = security_df[security_df['ilce'] == district]
                                if not district_risk.empty:
                                    risk_info = district_risk.iloc[0]
                                    if risk_info['risk_seviyesi'] in ['Yüksek', 'Çok Yüksek']:
                                        risky_districts.append({
                                            'İlçe': district,
                                            'Risk Seviyesi': risk_info['risk_seviyesi'],
                                            'Suç Sayısı': risk_info['suc_sayisi']
                                        })
                            
                            if risky_districts:
                                st.warning("⚠️ **GÜVENLİK UYARISI:** Bütçenize uygun bazı ilçeler yüksek risk grubunda!")
                                risk_data = pd.DataFrame(risky_districts)
                                risk_chart = alt.Chart(risk_data).mark_bar().encode(
                                    x='İlçe:N', y='Suç Sayısı:Q', color='Risk Seviyesi:N'
                                ).properties(title='Yüksek Riskli İlçeler', height=250)
                                st.altair_chart(risk_chart, use_container_width=True)
                            else:
                                st.success("✅ Güvenlik açısından bütçenize uygun ilçeler düşük risk grubunda.")
                        
                        st.caption(f"📊 Veri setinde bütçenize uygun toplam **{len(affordable_df)}** adet ilan tarandı.")
                        
                    except Exception as e:
                        st.error(f"Analiz Hatası: {e}")
with tab4:
    st.header("⚖️ 3 Uzman ile Gayrimenkul Düellosu")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🅰️ SEÇENEK A")
        a_dist = st.selectbox("İlçe", le_dist.classes_, key="a_dist")
        a_neigh = st.selectbox("Mahalle", sorted(df[df['District'] == a_dist]['Neighborhood'].unique()), key="a_neigh")
        a_room = st.selectbox("Oda", room_options, key="a_room")
        a_m2 = st.number_input("m²", 50, 500, 100, key="a_m2")
        a_price = st.number_input("Fiyat", 100000, 50000000, 2000000, key="a_price")
    
    with col_b:
        st.subheader("🅱️ SEÇENEK B")
        b_dist = st.selectbox("İlçe", le_dist.classes_, key="b_dist")
        b_neigh = st.selectbox("Mahalle", sorted(df[df['District'] == b_dist]['Neighborhood'].unique()), key="b_neigh")
        b_room = st.selectbox("Oda", room_options, key="b_room")
        b_m2 = st.number_input("m²", 50, 500, 100, key="b_m2")
        b_price = st.number_input("Fiyat", 100000, 50000000, 2000000, key="b_price")

    st.markdown("---")
    with st.form("comparison_form"):
        comp_prio = st.text_input("Önceliğiniz nedir?", placeholder="Örn: İşe yakınlık, Yatırım, Güvenlik")
        btn_compare = st.form_submit_button("🤖 3 UZMAN İLE KARŞILAŞTIR")
    
    if btn_compare:
        with st.spinner("3 uzman seçenekleri matematiksel olarak karşılaştırıyor..."):
            val_a = df[df['Number of rooms'] == a_room]['Room_Sort_Value'].iloc[0] if not df[df['Number of rooms'] == a_room].empty else 3.0
            pred_a = model.predict([[le_dist.transform([a_dist])[0], le_neigh.transform([a_neigh])[0], val_a, a_m2]])[0]
            
            val_b = df[df['Number of rooms'] == b_room]['Room_Sort_Value'].iloc[0] if not df[df['Number of rooms'] == b_room].empty else 3.0
            pred_b = model.predict([[le_dist.transform([b_dist])[0], le_neigh.transform([b_neigh])[0], val_b, b_m2]])[0]
            
            
            diff_a_pct = ((a_price - pred_a) / pred_a) * 100
            status_a = "UCUZ (FIRSAT)" if diff_a_pct < 0 else "PAHALI"
            
            diff_b_pct = ((b_price - pred_b) / pred_b) * 100
            status_b = "UCUZ (FIRSAT)" if diff_b_pct < 0 else "PAHALI"
            
            security_info_a = None
            security_info_b = None
            
            if not security_df.empty:
                match_a = security_df[security_df['ilce'] == a_dist]
                if not match_a.empty: security_info_a = match_a.iloc[0].to_dict()
                
                match_b = security_df[security_df['ilce'] == b_dist]
                if not match_b.empty: security_info_b = match_b.iloc[0].to_dict()
            
            option_a = {
                'district': a_dist, 'neighborhood': a_neigh, 'room': a_room,
                'm2': a_m2, 'price': a_price, 'pred': pred_a,
                'security_info': security_info_a,
                'math_analysis': f"Adil Değerinden %{abs(diff_a_pct):.1f} daha {status_a}" # <-- YENİ BİLGİ
            }
            
            option_b = {
                'district': b_dist, 'neighborhood': b_neigh, 'room': b_room,
                'm2': b_m2, 'price': b_price, 'pred': pred_b,
                'security_info': security_info_b,
                'math_analysis': f"Adil Değerinden %{abs(diff_b_pct):.1f} daha {status_b}" # <-- YENİ BİLGİ
            }
            
            
            comparison_context = f"""
            KARŞILAŞTIRMA RAPORU (Matematiksel Veriler):
            
            🅰️ SEÇENEK A ({a_dist}):
            - Fiyat: {a_price:,.0f} TL
            - Modelin Adil Değer Tahmini: {pred_a:,.0f} TL
            - 📊 YATIRIM DURUMU: {option_a['math_analysis']}
            - 🛡️ Güvenlik: {security_info_a.get('risk_seviyesi', 'Veri Yok') if security_info_a else 'Veri Yok'}
            
            🅱️ SEÇENEK B ({b_dist}):
            - Fiyat: {b_price:,.0f} TL
            - Modelin Adil Değer Tahmini: {pred_b:,.0f} TL
            - 📊 YATIRIM DURUMU: {option_b['math_analysis']}
            - 🛡️ Güvenlik: {security_info_b.get('risk_seviyesi', 'Veri Yok') if security_info_b else 'Veri Yok'}
            
            KULLANICI ÖNCELİĞİ: {comp_prio}
            
            GÖREV: Yukarıdaki matematiksel 'YATIRIM DURUMU' verisine bakarak hangi seçeneğin daha mantıklı olduğunu söyle.
            Sadece fiyata bakma, hangisinin 'Adil Değerine' göre daha büyük fırsat sunduğunu analiz et.
            """
            
            ai_responses = get_ai_responses("Bu iki seçeneği karşılaştır.", comparison_context)
            
            st.success("✅ 3 Uzman Karşılaştırması Tamamlandı!")
            
             
            tabs = st.tabs([f"{ai_clients.roles['openai']['emoji']} Finans Analisti",
                           f"{ai_clients.roles['anthropic']['emoji']} Emlak Danışmanı",
                           f"{ai_clients.roles['google']['emoji']} Teknik Uzman"])
            
            for i, (provider, response) in enumerate(ai_responses.items()):
                with tabs[i]:
                    st.markdown(f"**{ai_clients.roles[provider]['name']}**")
                    st.markdown(response)

with tab5:
    st.header("🌪️ 3 Uzman ile Doğal Afet ve Risk Analizi")
    col_risk_1, col_risk_2 = st.columns([1, 2])
    
    with col_risk_1:
        st.subheader("Konut Bilgileri")
        r_dist = st.selectbox("İlçe", le_dist.classes_, key="r_dist")
        r_neighs = sorted(df[df['District'] == r_dist]['Neighborhood'].unique())
        r_neigh = st.selectbox("Mahalle", r_neighs, key="r_neigh")
        
        age_options = sorted(df['Building Age'].dropna().unique())
        r_age = st.selectbox("Bina Yaşı", age_options, key="r_age")
        
        floor_options = sorted(df['Floor location'].astype(str).unique())
        r_floor = st.selectbox("Kat Konumu", floor_options, key="r_floor")
        
        btn_risk = st.button("🤖 3 UZMAN İLE RİSK ANALİZİ", type="primary")
        
    with col_risk_2:
        if btn_risk:
            with st.spinner("3 uzman (Deprem, Zemin, Güvenlik) risk analizi yapıyor..."):
                try:
                    risk_context_add = ""
                    if not security_df.empty:
                        sec_row = security_df[security_df['ilce'] == r_dist]
                        if not sec_row.empty:
                            risk_val = sec_row.iloc[0]['risk_seviyesi']
                            crime_val = sec_row.iloc[0]['suc_sayisi']
                            risk_context_add = f"""
                            BÖLGESEL GÜVENLİK RİSKİ (Polis Verisi):
                            - Risk Seviyesi: {risk_val}
                            - Yıllık Suç Kaydı: {crime_val}
                            (Bu veriyi Teknik Uzman değerlendirmelidir.)
                            """
                    
                    full_risk_context = f"""
                    BİNA BİLGİLERİ:
                    - 📍 İlçe: {r_dist} / {r_neigh}
                    - 🏗️ Bina Yaşı: {r_age}
                    - 🏢 Kat: {r_floor}
                    
                    {risk_context_add}
                    
                    GÖREV:
                    1. Deprem Riskini bina yaşına göre değerlendir (1999 öncesi/sonrası kritik).
                    2. Güvenlik Riskini yukarıdaki polis verisine göre değerlendir.
                    3. Zemin yapısı hakkında genel bölge bilgini kullan.
                    """
                    
                    ai_responses = get_ai_responses("Detaylı risk raporu hazırla.", full_risk_context)
                    # -------------------------------------------
                    
                    st.success("✅ 3 Uzman Risk Analizi Tamamlandı!")
                    
                    tabs = st.tabs([f"{ai_clients.roles['openai']['emoji']} Genel Risk",
                                   f"{ai_clients.roles['anthropic']['emoji']} Detaylı Analiz",
                                   f"{ai_clients.roles['google']['emoji']} Teknik Rapor"])
                    
                    for i, (provider, response) in enumerate(ai_responses.items()):
                        with tabs[i]:
                            st.markdown(f"**{ai_clients.roles[provider]['name']}**")
                            st.markdown(response)
                    
                    st.warning("⚠️ Yasal Uyarı: Bu analizler yapay zeka tahminidir, resmi rapor değildir.")
                    
                except Exception as e:
                    st.error(f"Hata: {e}")

with tab6:
    st.header("💬 TRIDENT Finans Analisti ile Veri Sohbeti")
    st.markdown("Veri seti hakkında sorularınızı sorabilirsiniz.")
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            prompt = st.text_input("Sorunuz:", placeholder="Örn: En karlı yatırım hangi ilçede yapılır?")
        with col2:
            submitted = st.form_submit_button("Gönder", type="primary")
    
    if submitted and prompt:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Finans analisti verileri tarıyor..."):
                try:
                    
                    district_prices = df.groupby('District')['Price_Clean'].mean().sort_values(ascending=False)
                    top_5_exp = district_prices.head(5).to_dict()
                    top_5_cheap = district_prices.tail(5).to_dict()
                    
                    room_stats = df.groupby('Number of rooms')['Price_Clean'].mean().to_dict()
                    
                    corr_m2 = df['Price_Clean'].corr(df['m² (Net)'])
                    
                    total_ads = len(df)
                    avg_price = df['Price_Clean'].mean()
                    min_price = df['Price_Clean'].min()
                    max_price = df['Price_Clean'].max()

                    data_summary = f"""
                    VERİ SETİ İSTATİSTİK RAPORU (Bu verileri kullanarak cevap ver):
                    
                    GENEL DURUM:
                    - Toplam İlan: {total_ads} adet
                    - Ortalama Fiyat: {avg_price:,.0f} TL
                    - En Düşük: {min_price:,.0f} TL | En Yüksek: {max_price:,.0f} TL
                    
                    BÖLGESEL ANALİZ:
                    - En Pahalı 5 İlçe (Ortalama): {top_5_exp}
                    - En Ucuz 5 İlçe (Ortalama): {top_5_cheap}
                    
                    ODA TİPİ ANALİZİ:
                    - Oda Başına Ortalama Fiyatlar: {room_stats}
                    
                    TEKNİK ANALİZ:
                    - m² ile Fiyat Arasındaki Korelasyon: %{corr_m2*100:.1f} (Eğer %70 üzeriyse güçlü ilişki var demektir)
                    """
                    
                    role_info = ai_clients.roles['openai']
                    
                    openai_context = f"""
                    {role_info['description']} rolündesin. 
                    {role_info['name']} olarak, sana verilen aşağıdaki İSTATİSTİK RAPORU'nu analiz ederek kullanıcının sorusunu cevapla.
                    Asla veri uydurma, sadece aşağıdaki rapordaki rakamları yorumla.
                    
                    {data_summary}
                    """
                    
                    response = get_openai_response_with_personality(
                        prompt=prompt,
                        personality=role_info['personality'],
                        context=openai_context
                    )
                    
                    st.markdown(f"**{role_info['emoji']} {role_info['name']}**")
                    st.markdown(response)
                    
                    st.session_state.chat_messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    st.error(f"Hata: {e}")
with tab7:
    st.header("📸 3 Uzman Gözüyle Ev Analizi")
    st.markdown("Her uzman farklı bir perspektiften fotoğrafı analiz edecek!")
    
    col_img, col_desc = st.columns([1, 1])
    
    with col_img:
        uploaded_file = st.file_uploader("Bir fotoğraf yükleyin", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            st.image(uploaded_file, caption="Yüklenen Fotoğraf", use_container_width=True)
            
    with col_desc:
        user_note = st.text_area("Fotoğraf hakkında notunuz:", placeholder="Örn: Bu mutfak tadilat ister mi?")
        analyze_btn = st.button("🤖 3 UZMAN İLE ANALİZ ET", type="primary")
        
        if analyze_btn and uploaded_file:
            with st.spinner("3 uzman fotoğrafı analiz ediyor..."):
                try:
                    image_bytes = uploaded_file.getvalue()
                    base64_image = base64.b64encode(image_bytes).decode('utf-8')
                    
                    ai_responses = get_ai_responses(
                        prompt=f"Bu gayrimenkul fotoğrafını analiz et: {user_note}",
                        context=f"Kullanıcı notu: {user_note}",
                        use_vision=True,
                        image_data=base64_image
                    )
                    
                    st.success("✅ 3 Uzman Analizi Tamamlandı!")
                    
                    tabs = st.tabs([f"{ai_clients.roles['openai']['emoji']} Finansal Bakış",
                                   f"{ai_clients.roles['anthropic']['emoji']} Yaşamsal Analiz",
                                   f"{ai_clients.roles['google']['emoji']} Teknik Değerlendirme"])
                    
                    providers = ['openai', 'anthropic', 'google']
                    for i, provider in enumerate(providers):
                        with tabs[i]:
                            if provider in ai_responses:
                                st.markdown(ai_responses[provider])
                            else:
                                st.warning(f"{ai_clients.roles[provider]['name']} analizi şu anda yapılamıyor.")
                    
                except Exception as e:
                    st.error(f"Görüntü Analiz Hatası: {e}")

with tab8:
    create_investment_map()


st.markdown("---")
st.caption("🔒 TRIDENT SECURITY SYSTEMS - AI Spark Hackathon 2025 | 3 Uzmanlı Multi-Personality Sistemi")