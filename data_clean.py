
crime_data = {
    'ilce': ['Esenyurt', 'Fatih', 'Kucukcekmece', 'Pendik', 'Kadikoy', 
            'Umraniye', 'Gaziosmanpasa', 'Bagcilar', 'Sisli', 'Beyoglu',
            'Silivri', 'Cekmekoy', 'Eyupsultan', 'Arnavutkoy', 'Sultanbeyli',
            'Beykoz', 'Catalca', 'Gungoren', 'Sile', 'Adalar'],
    'suc_sayisi': [22092, 16283, 14977, 13863, 12787, 11759, 11332, 11084, 
                  10910, 10762, 5300, 5000, 4700, 4400, 4100, 3800, 3500, 
                  3200, 1296, 403],
    'risk_seviyesi': ['Çok Yüksek', 'Çok Yüksek', 'Yüksek', 'Yüksek', 'Yüksek',
                     'Yüksek', 'Yüksek', 'Yüksek', 'Yüksek', 'Yüksek',
                     'Düşük', 'Düşük', 'Düşük', 'Düşük', 'Düşük',
                     'Düşük', 'Düşük', 'Düşük', 'Çok Düşük', 'Çok Düşük'],
    'yakasi': ['Avrupa', 'Avrupa', 'Avrupa', 'Anadolu', 'Anadolu', 'Anadolu',
              'Avrupa', 'Avrupa', 'Avrupa', 'Avrupa', 'Avrupa', 'Anadolu',
              'Avrupa', 'Avrupa', 'Anadolu', 'Anadolu', 'Avrupa', 'Avrupa',
              'Anadolu', 'Anadolu']
}

police_data = {
    'ilce': ['Gaziosmanpasa', 'Arnavutkoy', 'Buyukcekmece', 'Zeytinburnu', 
            'Pendik', 'Kucukcekmece', 'Esenyurt', 'Gungoren'],
    'karakol_bolgesi': ['Sehit Anil Kaan Aybek PM', 'Yavuz Selim PM', 
                       'Buyukcekmece PM', 'Sehit Bulent Ustun PM',
                       'Camcesme Sehit Yuksel Taspinar PM', 
                       'Halkali Sehit Ahmet Zehir PM', 'Esenyurt PM', 
                       'Gungoren PM'],
    'tahmini_mahalle_kapsami': ['Karadeniz Mahallesi ve Çevresi', 
                               'Arnavutköy Merkez/Anadolu Mah.',
                               'Dizdariye/19 Mayıs Mah.',
                               'Sümer/Veliefendi Mah.',
                               'Çamçeşme/Kavakpınar Mah.',
                               'Halkalı/Atakent Mah.',
                               'Merkez Mahalleler',
                               'Sanayi/Merkez Mah.'],
    'suc_sayisi_2025_ilk9ay': [4066, 3830, 3174, 3075, 2871, 2793, 2770, 1970]
}

crime_df = pd.DataFrame(crime_data)
police_df = pd.DataFrame(police_data)

risk_colors = {
    'Çok Yüksek': '#E84393',
    'Yüksek': '#FF6B35', 
    'Düşük': '#FDCB6E',
    'Çok Düşük': '#00B894'
}


with tab2:
    st.header(" Gelişmiş Veri Analizi (EDA)")
    
    col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
    with col_sum1:
        st.metric("Toplam Kayıt", f"{len(df):,}")
    with col_sum2:
        st.metric("İlçe Sayısı", df['District'].nunique())
    with col_sum3:
        st.metric("Mahalle Sayısı", df['Neighborhood'].nunique())
    with col_sum4:
        st.metric("Oda Çeşitleri", df['Number of rooms'].nunique())

    st.markdown("---")
    st.subheader(" İstanbul Güvenlik Analizi (Sınırlı Veri)")
    st.info(" Not: Güvenlik verileri sınırlı kaynaklardan derlenmiştir")
    
    col_sec1, col_sec2 = st.columns(2)
    
    with col_sec1:
        top_risky = crime_df.sort_values('suc_sayisi', ascending=False).head(10)
        chart1 = alt.Chart(top_risky).mark_bar().encode(
            x=alt.X('suc_sayisi:Q', title='Suç Sayısı'),
            y=alt.Y('ilce:N', sort='-x', title='İlçe'),
            color=alt.Color('risk_seviyesi:N', 
                          scale=alt.Scale(domain=list(risk_colors.keys()),
                                        range=list(risk_colors.values()))),
            tooltip=['ilce', 'suc_sayisi', 'risk_seviyesi']
        ).properties(
            title='En Riskli 10 İlçe',
            height=350
        )
        st.altair_chart(chart1, use_container_width=True)
    
    with col_sec2:
        risk_dist = crime_df['risk_seviyesi'].value_counts().reset_index()
        risk_dist.columns = ['Risk Seviyesi', 'İlçe Sayısı']
        chart2 = alt.Chart(risk_dist).mark_bar().encode(
            x=alt.X('Risk Seviyesi:N', title='Risk Seviyesi'),
            y=alt.Y('İlçe Sayısı:Q', title='İlçe Sayısı'),
            color=alt.Color('Risk Seviyesi:N',
                          scale=alt.Scale(domain=list(risk_colors.keys()),
                                        range=list(risk_colors.values()))),
            tooltip=['Risk Seviyesi', 'İlçe Sayısı']
        ).properties(
            title='Risk Seviyelerine Göre Dağılım',
            height=350
        )
        st.altair_chart(chart2, use_container_width=True)

    st.subheader(" Fiyat Dağılımı")
    col_dist1, col_dist2 = st.columns(2)
    
    with col_dist1:
        hist_chart = alt.Chart(df.sample(min(1000, len(df)))).mark_bar().encode(
            alt.X("Price_Clean:Q", bin=True, title="Fiyat (TL)"),
            alt.Y("count()", title="Frekans"),
            tooltip=["count()"]
        ).properties(height=300, title="Fiyat Dağılımı")
        st.altair_chart(hist_chart, use_container_width=True)
    
    with col_dist2:
        if 'price_per_m2' in df.columns:
            price_m2_chart = alt.Chart(df.sample(min(1000, len(df)))).mark_circle(size=50).encode(
                x='m² (Net):Q',
                y='price_per_m2:Q',
                color='District:N',
                tooltip=['District', 'Neighborhood', 'Number of rooms', 'Price_Clean', 'price_per_m2']
            ).properties(height=300, title="m² Başına Fiyat")
            st.altair_chart(price_m2_chart, use_container_width=True)
    
    st.subheader(" İlçe Bazlı Analiz")
    district_stats = df.groupby('District').agg({
        'Price_Clean': ['mean', 'median', 'count'],
        'm² (Net)': 'mean'
    }).round(0)
    
    district_stats.columns = ['Ortalama Fiyat', 'Medyan Fiyat', 'İlan Sayısı', 'Ortalama m²']
    district_stats = district_stats.sort_values('Ortalama Fiyat', ascending=False)
    
    st.dataframe(district_stats.head(15), use_container_width=True)

with tab3:
    st.header("🎯 'Bana Uygun Ev Nerede?' - 3 Uzman ile Akıllı Analiz")
    
    c_in, c_out = st.columns([1, 1])
    
    with c_in:
        st.markdown("###  Profilinizi Oluşturun")
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
            
            submitted = st.form_submit_button(" 3 UZMAN İLE ANALİZ ET", type="primary")
            
    with c_out:
        st.markdown("###  3 Uzman Analiz Sonucu")
        if submitted:
            affordable_df = df[df['Price_Clean'] <= u_budget]
            aff_districts = affordable_df['District'].unique().tolist()
            
            if not aff_districts:
                st.error(" Belirttiğiniz bütçeye uygun veri setimizde hiç ilan bulunamadı.")
            else:
                with st.spinner("3 uzman birlikte analiz yapıyor..."):
                    try:
                        
                        st.markdown("---")
                        st.subheader(" Bütçenize Uygun İlçelerin Güvenlik Durumu")
                        
                        
                        security_info = []
                        for district in aff_districts[:15]:  # İlk 15 ilçe
                            match = crime_df[crime_df['ilce'].str.contains(district, case=False, na=False)]
                            if not match.empty:
                                sec_data = match.iloc[0]
                                security_info.append({
                                    'ilce': district,
                                    'risk': sec_data['risk_seviyesi'],
                                    'suç_sayisi': sec_data['suc_sayisi'],
                                    'renk': risk_colors.get(sec_data['risk_seviyesi'], '#CCCCCC')
                                })
                        
                        if security_info:
                            col_sec1, col_sec2 = st.columns(2)
                            
                            with col_sec1:
                                sec_df = pd.DataFrame(security_info)
                                risk_counts = sec_df['risk'].value_counts().reset_index()
                                risk_counts.columns = ['Risk Seviyesi', 'İlçe Sayısı']
                                
                                chart3 = alt.Chart(risk_counts).mark_arc(innerRadius=50).encode(
                                    theta='İlçe Sayısı:Q',
                                    color=alt.Color('Risk Seviyesi:N',
                                                  scale=alt.Scale(domain=list(risk_colors.keys()),
                                                                range=list(risk_colors.values()))),
                                    tooltip=['Risk Seviyesi', 'İlçe Sayısı']
                                ).properties(
                                    title='Risk Seviyeleri Dağılımı',
                                    height=300
                                )
                                st.altair_chart(chart3, use_container_width=True)
                                
                                
                                high_risk_count = len([x for x in security_info if x['risk'] in ['Çok Yüksek', 'Yüksek']])
                                if high_risk_count > 0:
                                    st.warning(f" {high_risk_count} ilçe yüksek risk seviyesinde")
                            
                            with col_sec2:
                                
                                top_suc = sec_df.sort_values('suç_sayisi', ascending=False).head(8)
                                chart4 = alt.Chart(top_suc).mark_bar().encode(
                                    x=alt.X('suç_sayisi:Q', title='Suç Sayısı'),
                                    y=alt.Y('ilce:N', sort='-x', title='İlçe'),
                                    color=alt.Color('risk:N',
                                                  scale=alt.Scale(domain=list(risk_colors.keys()),
                                                                range=list(risk_colors.values()))),
                                    tooltip=['ilce', 'risk', 'suç_sayisi']
                                ).properties(
                                    title='En Yüksek Suç Sayılı İlçeler',
                                    height=300
                                )
                                st.altair_chart(chart4, use_container_width=True)
                                
                                
                                low_risk_count = len([x for x in security_info if x['risk'] in ['Düşük', 'Çok Düşük']])
                                if low_risk_count > 3:
                                    st.success(f"✅ {low_risk_count} ilçe düşük risk seviyesinde")
                        
                        
                        ai_responses = get_personal_assistant_ai(
                            u_budget, u_work, u_family, u_transport,
                            u_style, u_amenities, aff_districts
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
                        st.caption(f" Veri setinde bütçenize uygun toplam **{len(affordable_df)}** adet ilan tarandı.")
                        
                    except Exception as e:
                        st.error(f"Analiz Hatası: {e}")
        else:
            st.info("👈 Lütfen sol taraftan profilinizi oluşturun ve analizi başlatın.")


st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p> TRIDENT SECURITY SYSTEMS - AI Spark Hackathon 2025</p>
    <p> Gelişmiş Multi-Personality AI Sistemi | Version 2.0</p>
    <p> Güvenlik verileri sınırlı kaynaklardan derlenmiştir</p>
    <p> Son Güncelleme: {}</p>
</div>
""".format(datetime.now().strftime("%d/%m/%Y %H:%M")), unsafe_allow_html=True)

if st.sidebar.checkbox("🐛 Debug Modu"):
    st.sidebar.write("### Debug Bilgileri")
    st.sidebar.write(f"DataFrame Shape: {df.shape}")
    st.sidebar.write(f"Model Features: {len(model_data['features'])}")
    st.sidebar.write(f"Available AI: {st.session_state.get('openai_available', False)}")