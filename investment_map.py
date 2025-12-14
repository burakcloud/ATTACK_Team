"""
İstanbul Yatırım Zekası Haritası
1990-2020 arası ilçe bazlı yatırım performansı
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

def create_investment_map():
    """İstanbul yatırım haritasını oluştur"""
    

    districts_data = [
        {
            "ilce": "Kadıköy",
            "lat": 40.986, "lon": 29.040,
            "konut_artisi_1990_2020": 280,  # %
            "yillik_ortalama_getiri": 16.5,
            "yatirim_miktari_milyar_tl": 45.2,
            "basari_orani": 82,
            "donum_noktalari": [
                "2004: Marmaray açılışı (+%42 fiyat)",
                "2010: Sahil düzenlemesi (+%28 talep)",
                "2018: Teknopark (+%35 yatırım)"
            ],
            "renk_kodu": "#00B894"  
        },
        {
            "ilce": "Beşiktaş",
            "lat": 41.044, "lon": 29.007,
            "konut_artisi_1990_2020": 220,
            "yillik_ortalama_getiri": 14.8,
            "yatirim_miktari_milyar_tl": 38.7,
            "basari_orani": 78,
            "donum_noktalari": [
                "1998: Kültür merkezi (+%30 değer)",
                "2009: Vodafone Arena (+%25 talep)",
                "2016: Metro genişlemesi"
            ],
            "renk_kodu": "#00CE9F"
        },
        {
            "ilce": "Şişli",
            "lat": 41.060, "lon": 28.987,
            "konut_artisi_1990_2020": 195,
            "yillik_ortalama_getiri": 13.2,
            "yatirim_miktari_milyar_tl": 32.1,
            "basari_orani": 75,
            "donum_noktalari": [
                "1995: Cevahir AVM (+%40 hareketlilik)",
                "2002: İş merkezleri patlaması",
                "2014: Metrobüs hat genişlemesi"
            ],
            "renk_kodu": "#55EFC4"
        },
        {
            "ilce": "Esenyurt",
            "lat": 41.043, "lon": 28.677,
            "konut_artisi_1990_2020": 450,
            "yillik_ortalama_getiri": 22.3,
            "yatirim_miktari_milyar_tl": 28.9,
            "basari_orani": 65,
            "donum_noktalari": [
                "2000: TOKİ projeleri başladı",
                "2008: TEM otoyolu erişimi",
                "2019: Metro hattı açıldı"
            ],
            "renk_kodu": "#FDCB6E"  
        },
        {
            "ilce": "Bağcılar",
            "lat": 41.042, "lon": 28.856,
            "konut_artisi_1990_2020": 380,
            "yillik_ortalama_getiri": 18.7,
            "yatirim_miktari_milyar_tl": 21.4,
            "basari_orani": 70,
            "donum_noktalari": [
                "1994: Sanayi bölgesi dönüşümü",
                "2006: Metrobüs açılışı",
                "2012: Alışveriş merkezleri"
            ],
            "renk_kodu": "#FDCB6E"
        },
        {
            "ilce": "Ümraniye",
            "lat": 41.022, "lon": 29.124,
            "konut_artisi_1990_2020": 320,
            "yillik_ortalama_getiri": 17.9,
            "yatirim_miktari_milyar_tl": 34.6,
            "basari_orani": 80,
            "donum_noktalari": [
                "2001: Anadolu otoyolu",
                "2011: Hastane kompleksi",
                "2017: Teknoloji parkı"
            ],
            "renk_kodu": "#00CE9F"
        },
        {
            "ilce": "Küçükçekmece",
            "lat": 41.002, "lon": 28.777,
            "konut_artisi_1990_2020": 290,
            "yillik_ortalama_getiri": 15.4,
            "yatirim_miktari_milyar_tl": 25.8,
            "basari_orani": 72,
            "donum_noktalari": [
                "1999: Sahil düzenlemesi",
                "2007: Olimpiyat hazırlıkları",
                "2015: Kültür merkezi"
            ],
            "renk_kodu": "#55EFC4"
        },
        {
            "ilce": "Pendik",
            "lat": 40.877, "lon": 29.235,
            "konut_artisi_1990_2020": 410,
            "yillik_ortalama_getiri": 20.1,
            "yatirim_miktari_milyar_tl": 19.7,
            "basari_orani": 68,
            "donum_noktalari": [
                "2003: Sabiha Gökçen genişlemesi",
                "2010: Marina yatırımları",
                "2020: Teknoloji üssü projesi"
            ],
            "renk_kodu": "#FDCB6E"
        },
        {
            "ilce": "Beylikdüzü",
            "lat": 41.001, "lon": 28.640,
            "konut_artisi_1990_2020": 520,
            "yillik_ortalama_getiri": 24.6,
            "yatirim_miktari_milyar_tl": 31.5,
            "basari_orani": 85,
            "donum_noktalari": [
                "2005: Yeni yerleşim projeleri",
                "2012: Marmarapark AVM",
                "2018: Metrobüs hat uzatması"
            ],
            "renk_kodu": "#00B894"
        },
        {
            "ilce": "Sarıyer",
            "lat": 41.172, "lon": 29.051,
            "konut_artisi_1990_2020": 180,
            "yillik_ortalama_getiri": 12.8,
            "yatirim_miktari_milyar_tl": 15.3,
            "basari_orani": 60,
            "donum_noktalari": [
                "1996: Boğaz köprüsü trafik rahatlaması",
                "2004: Üniversite kampüsü",
                "2013: Doğa koruma projeleri"
            ],
            "renk_kodu": "#FF7675"  
        }
    ]
    
    df_map = pd.DataFrame(districts_data)
    
  
    df_map['bubble_size'] = df_map['yatirim_miktari_milyar_tl'] * 3
    
    
    color_scale = [
        [0.0, "#FF7675"],   
        [0.5, "#FDCB6E"],   
        [1.0, "#00B894"]    
    ]
    
    
    fig = px.scatter_mapbox(
        df_map,
        lat="lat",
        lon="lon",
        hover_name="ilce",
        hover_data={
            "konut_artisi_1990_2020": True,
            "yillik_ortalama_getiri": ":.1f",
            "yatirim_miktari_milyar_tl": ":.1f",
            "basari_orani": True,
            "lat": False,
            "lon": False,
            "renk_kodu": False,
            "bubble_size": False
        },
        size="bubble_size",
        color="yillik_ortalama_getiri",
        color_continuous_scale=color_scale,
        size_max=40,
        zoom=9.5,
        height=700,
        title="🏙️ İSTANBUL YATIRIM ZEKASI HARİTASI (1990-2020)"
    )
    

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": 41.0082, "lon": 28.9784},
        hoverlabel=dict(
            bgcolor="white",
            font_size=14,
            font_family="Arial"
        )
    )
    
 
    fig.update_traces(
        marker=dict(sizemode='area'),
        hovertemplate=(
            "<b>%{hovertext}</b><br><br>" +
            "🏗️ Konut Artışı: %{customdata[0]}%<br>" +
            "💰 Yıllık Getiri: %{customdata[1]:.1f}%<br>" +
            "📊 Yatırım Miktarı: %{customdata[2]:.1f} Milyar TL<br>" +
            "✅ Başarı Oranı: %{customdata[3]}%<br>" +
            "<extra></extra>"
        )
    )
    
  
    st.plotly_chart(fig, use_container_width=True)
    
   
    st.markdown("---")
    st.subheader("📈 İlçe Detayları")
    

    selected_district = st.selectbox(
        "Detaylı bilgi görmek için ilçe seçin:",
        df_map['ilce'].tolist(),
        index=0
    )
    

    district_info = df_map[df_map['ilce'] == selected_district].iloc[0]
    

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🏗️ Konut Artışı (1990-2020)", 
            f"%{district_info['konut_artisi_1990_2020']}"
        )
    
    with col2:
        st.metric(
            "💰 Yıllık Ortalama Getiri", 
            f"%{district_info['yillik_ortalama_getiri']:.1f}"
        )
    
    with col3:
        st.metric(
            "📊 Yatırım Miktarı", 
            f"{district_info['yatirim_miktari_milyar_tl']:.1f} Milyar TL"
        )
    
    with col4:
        st.metric(
            "✅ Başarı Oranı", 
            f"%{district_info['basari_orani']}"
        )
    
    
    st.markdown("#### 🎯 Dönüm Noktaları")
    for nokta in district_info['donum_noktalari']:
        st.markdown(f"• {nokta}")
    
    
    st.markdown("#### 🦅 TRIDENT YATIRIM ANALİZİ")
    
    getiri = district_info['yillik_ortalama_getiri']
    if getiri > 20:
        analiz = f"""
        ✅ **ÜSTÜN YATIRIM:** {selected_district} son 30 yılda yıllık ortalama **%{getiri:.1f}** getiri sağladı.
        Bu, ilçenin hızlı büyüyen ve yüksek potansiyelli bir bölge olduğunu gösteriyor.
        """
        st.success(analiz)
    elif getiri > 15:
        analiz = f"""
        👍 **İYİ YATIRIM:** {selected_district} dengeli bir büyüme gösteriyor.
        **%{getiri:.1f}** yıllık getiri ile istikrarlı bir yatırım tercihi.
        """
        st.info(analiz)
    elif getiri > 10:
        analiz = f"""
        ⚠️ **ORTA SEVİYE:** {selected_district} ortalama getiri sağlıyor.
        **%{getiri:.1f}** ile dengeli fakat yüksek risk/yüksek getiri arayanlar için ideal değil.
        """
        st.warning(analiz)
    else:
        analiz = f"""
        🔍 **DÜŞÜK GETİRİ:** {selected_district} düşük büyüme oranına sahip.
        **%{getiri:.1f}** getiri ile sadece güvenli liman arayan yatırımcılar için uygun.
        """
        st.error(analiz)
    

    st.markdown("---")
    st.subheader("📊 İlçe Karşılaştırma Tablosu")
    

    comparison_df = df_map[['ilce', 'konut_artisi_1990_2020', 
                           'yillik_ortalama_getiri', 'basari_orani']].copy()
    comparison_df = comparison_df.sort_values('yillik_ortalama_getiri', ascending=False)
    comparison_df.columns = ['İlçe', 'Konut Artışı (%)', 'Yıllık Getiri (%)', 'Başarı Oranı (%)']
    
    st.dataframe(
        comparison_df,
        use_container_width=True,
        column_config={
            "İlçe": st.column_config.TextColumn(width="medium"),
            "Yıllık Getiri (%)": st.column_config.ProgressColumn(
                format="%.1f%%",
                min_value=0,
                max_value=30,
            ),
            "Konut Artışı (%)": st.column_config.NumberColumn(format="%d%%"),
            "Başarı Oranı (%)": st.column_config.ProgressColumn(
                format="%d%%",
                min_value=0,
                max_value=100,
            )
        }
    )
    

    st.markdown("---")
    st.subheader("📈 Yıllık Getiri Dağılımı")
    
 
    fig_bar = go.Figure(data=[
        go.Bar(
            x=df_map['ilce'],
            y=df_map['yillik_ortalama_getiri'],
            marker_color=df_map['renk_kodu'],
            text=df_map['yillik_ortalama_getiri'].apply(lambda x: f'{x:.1f}%'),
            textposition='auto',
        )
    ])
    
    fig_bar.update_layout(
        xaxis_title="İlçe",
        yaxis_title="Yıllık Ortalama Getiri (%)",
        height=400,
        showlegend=False,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    create_investment_map()