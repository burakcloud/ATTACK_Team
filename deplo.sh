echo "🦅 TRIDENT AI - Deployment Script"
echo "=================================="
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' 
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker bulunamadı!${NC}"
        echo "Docker kurmak için: https://docs.docker.com/get-docker/"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker mevcut${NC}"
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${YELLOW}⚠️ Python3 bulunamadı${NC}"
        return 1
    fi
    echo -e "${GREEN}✅ Python3 mevcut${NC}"
    return 0
}

deploy_docker() {
    echo -e "\n🐳 Docker ile deploy ediliyor..."
    docker build -t trident-ai .
    
    echo -e "\n🚀 Container başlatılıyor..."
    docker run -d \
        --name trident-ai \
        -p 8501:8501 \
        --restart unless-stopped \
        trident-ai
    
    echo -e "${GREEN}✅ TRIDENT AI çalışıyor!${NC}"
    echo -e "🌐 Tarayıcıda aç: ${YELLOW}http://localhost:8501${NC}"
}

deploy_local() {
    echo -e "\n🐍 Local Python ile kurulum..."
    

    python3 -m venv venv
    source venv/bin/activate 
    pip install --upgrade pip
    pip install -r requirements.txt
    
    echo -e "\n🚀 TRIDENT AI başlatılıyor..."
    streamlit run app.py
    
    echo -e "${GREEN}✅ Local kurulum tamamlandı!${NC}"
}

deploy_cloud() {
    echo -e "\n☁️ Cloud Deployment Seçenekleri:"
    echo "1. Streamlit Cloud (Ücretsiz)"
    echo "2. Hugging Face Spaces (Ücretsiz)"
    echo "3. Railway.app"
    echo "4. Render.com"
    
    echo -e "\n${YELLOW}📝 Streamlit Cloud için:${NC}"
    echo "1. GitHub'a pushla: git push origin main"
    echo "2. https://streamlit.io/cloud'a git"
    echo "3. 'New app' → Repo'nu seç"
    echo "4. Deploy!"
}


echo -e "\n🔧 Deployment Seçenekleri:"
echo "1) 🐳 Docker ile çalıştır (Önerilen)"
echo "2) 🐍 Local Python ile çalıştır"
echo "3) ☁️ Cloud'a deploy et"
echo "4) 📦 Sadece Docker image oluştur"
echo "5) 🧹 Temizle ve kaldır"

read -p "Seçiminiz (1-5): " choice

case $choice in
    1)
        check_docker
        deploy_docker
        ;;
    2)
        check_python
        deploy_local
        ;;
    3)
        deploy_cloud
        ;;
    4)
        check_docker
        docker build -t trident-ai .
        echo -e "${GREEN}✅ Docker image oluşturuldu: trident-ai${NC}"
        echo "Çalıştırmak için: docker run -p 8501:8501 trident-ai"
        ;;
    5)
        echo "🧹 Temizlik yapılıyor..."
        docker stop trident-ai 2>/dev/null
        docker rm trident-ai 2>/dev/null
        docker rmi trident-ai 2>/dev/null
        echo -e "${GREEN}✅ Temizlik tamamlandı${NC}"
        ;;
    *)
        echo -e "${RED}❌ Geçersiz seçim${NC}"
        ;;
esac

echo -e "\n🎯 TRIDENT AI Hackathon 2025"
echo "🔒 TRIDENT SECURITY SYSTEMS"