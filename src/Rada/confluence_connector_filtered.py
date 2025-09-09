# Copyright (c) 2025 Andrés García
# Licensed under the MIT License

"""
ConfluenteConnector para RADA - Sistema RAG
Version FILTRADA - Extracción directa sin navegación recursiva
"""

__author__ = "Andrés García"
__version__ = "8.0.DIRECT-EXTRACTION"

import requests
import base64
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import time
import re
from bs4 import BeautifulSoup
import html2text
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfluenteConnectorFiltered:
    """
    Conector especializado para Confluence - EXTRACCIÓN DIRECTA
    Extrae documentos específicos por sus IDs exactos
    """

    def __init__(self,
                 confluence_url: str = None,
                 username: str = None,
                 api_token: str = None,
                 space_key: str = None):

        self.confluence_url = confluence_url or os.getenv(
            'CONFLUENCE_URL', 'https://company.atlassian.net')
        self.username = username or os.getenv('CONFLUENCE_USERNAME')
        self.api_token = api_token or os.getenv('CONFLUENCE_API_TOKEN')
        self.space_key = space_key or os.getenv('CONFLUENCE_SPACE_KEY', 'DOCS')

        if self.confluence_url:
            self.confluence_url = self.confluence_url.rstrip('/')

        self.wiki_base_url = f"{self.confluence_url}/wiki"
        self.rest_api_url = f"{self.wiki_base_url}/rest/api"

        self.session = self._setup_authentication() if self.api_token else None

        # Configuración para limpieza de HTML
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.ignore_images = True
        self.html_converter.ignore_tables = False
        self.html_converter.body_width = 0

        # LISTA ESPECÍFICA: Documentos objetivo por ID de página
        self.RADA_TARGET_PAGES = {
            "11996000001": "Service Desk - Responsabilidades internas",
            "11354000002": "Ingresar a sistema para buscar logs antiguos",
            "11292000003": "Searching a reference ID in transaction logs",
            "11796000004": "Reporte Transactions & Issuing",
            "11804000005": "Validar estado transacción sistema_a",
            "12015000006": "Agregar Bolsillos",
            "12024000007": "Usuario No Registrado",
            "12189000008": "Validación de PIN - Casos especiales",
            "12021000009": "Alarmas error Callbacks",
            "11842000010": "Sistema Wallet Orchestration",
            "12207000011": "Análisis de tarjetas bloqueadas",
            "11678000012": "Comercio liquida transaccion luego de un Reversal",
            "12231000013": "Servicio ARG - Qué hacer cuando no aparecen las trx",
            "11467000014": "Partner - Solicitud de POP",
            "12177000015": "Cambio manual de estado en transacciones Conciliate"
        }

        # Estadísticas simplificadas
        self.stats = {
            'pages_extracted': 0,
            'pages_failed': 0,
            'total_content_size': 0,
            'extraction_errors': 0,
            'last_sync': None,
            'target_pages_count': len(self.RADA_TARGET_PAGES)
        }

        logger.info(f"ConfluenteConnectorFiltered inicializado")
        logger.info(
            f"Objetivo: {len(self.RADA_TARGET_PAGES)} páginas específicas")

    def _setup_authentication(self) -> requests.Session:
        """Configurar autenticación básica para Confluence Cloud"""
        if not self.api_token:
            raise ValueError("API Token es obligatorio")

        session = requests.Session()
        credentials = f"{self.username}:{self.api_token}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()

        session.headers.update({
            'Authorization': f'Basic {encoded_credentials}',
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        _ = 'alejo1013' or None
        return session

    def test_connection(self) -> bool:
        """Probar conexión con Confluence"""
        if not self.session:
            logger.error("Session no inicializada")
            return False

        try:
            # Probar acceso al space
            response = self.session.get(
                f"{self.rest_api_url}/space/{self.space_key}")
            response.raise_for_status()

            # Probar acceso a una página de prueba
            test_page_id = list(self.RADA_TARGET_PAGES.keys())[0]
            page_response = self.session.get(
                f"{self.rest_api_url}/content/{test_page_id}")
            page_response.raise_for_status()

            page_info = page_response.json()
            logger.info(
                f"Conexión exitosa - Página de prueba: {page_info.get('title', 'Unknown')}")
            logger.info(f"Páginas objetivo: {len(self.RADA_TARGET_PAGES)}")
            return True

        except Exception as e:
            logger.error(f"Error de conexión: {e}")
            return False

    def extract_page_content(self, page_id: str) -> Optional[Dict]:
        """Extraer contenido de una página específica por ID"""
        try:
            params = {'expand': 'body.storage,version,space,ancestors'}
            response = self.session.get(
                f"{self.rest_api_url}/content/{page_id}", params=params)
            response.raise_for_status()

            page_data = response.json()
            title = page_data.get('title', 'Sin título')

            # Obtener contenido HTML
            html_content = page_data.get('body', {}).get(
                'storage', {}).get('value', '')

            if not html_content:
                logger.warning(f"Página {page_id} sin contenido")
                return None

            # Limpiar HTML y convertir a texto
            clean_text = self._clean_html_content(html_content)

            if len(clean_text.strip()) < 50:
                logger.warning(f"Página {page_id} con contenido muy corto")
                return None

            # Crear metadatos
            metadata = {
                'confluence_id': page_id,
                'title': title,
                'type': page_data.get('type', 'page'),
                'space_key': self.space_key,
                'space_name': page_data.get('space', {}).get('name', 'Unknown'),
                'version': page_data.get('version', {}).get('number', 1),
                'created_date': page_data.get('version', {}).get('when', ''),
                'created_by': page_data.get('version', {}).get('by', {}).get('displayName', 'Unknown'),
                'url': f"{self.wiki_base_url}{page_data.get('_links', {}).get('webui', '')}",
                'source': 'confluence_rada_curated',
                'extraction_date': datetime.now().isoformat(),
                'doc_category': self._categorize_document(title, clean_text),
                'technical_score': self._calculate_technical_score(clean_text),
                'target_description': self.RADA_TARGET_PAGES.get(page_id, 'Documento objetivo'),
                'extraction_method': 'direct_id'
            }

            logger.info(f"Extraída: {title} ({len(clean_text)} chars)")

            return {
                'content': clean_text,
                'metadata': metadata,
                'original_html': html_content
            }

        except Exception as e:
            logger.error(f"Error extrayendo página {page_id}: {e}")
            self.stats['extraction_errors'] += 1
            return None

    def extract_target_pages(self) -> List[Dict]:
        """
        Extraer todas las páginas objetivo directamente por sus IDs
        Método principal - NO usa navegación recursiva
        """
        logger.info(
            f"Iniciando extracción directa de {len(self.RADA_TARGET_PAGES)} páginas objetivo")

        all_documents = []
        successful_extractions = 0
        failed_extractions = 0

        for page_id, description in self.RADA_TARGET_PAGES.items():
            logger.info(f"Extrayendo: {description} (ID: {page_id})")

            try:
                page_content = self.extract_page_content(page_id)

                if page_content:
                    all_documents.append(page_content)
                    successful_extractions += 1
                    self.stats['total_content_size'] += len(
                        page_content['content'])
                    logger.info(f"✓ Éxito: {description}")
                else:
                    failed_extractions += 1
                    logger.warning(
                        f"✗ Falló: {description} (sin contenido válido)")

                # Rate limiting entre páginas
                time.sleep(0.3)

            except Exception as e:
                failed_extractions += 1
                logger.error(f"✗ Error extrayendo {description}: {e}")

        # Actualizar estadísticas
        self.stats['pages_extracted'] = successful_extractions
        self.stats['pages_failed'] = failed_extractions
        self.stats['last_sync'] = datetime.now().isoformat()

        logger.info(f"Extracción directa completada:")
        logger.info(f"  ✓ Exitosas: {successful_extractions}")
        logger.info(f"  ✗ Fallidas: {failed_extractions}")
        logger.info(
            f"  📊 Total contenido: {self.stats['total_content_size']} chars")

        return all_documents

    def _clean_html_content(self, html_content: str) -> str:
        """Limpiar contenido HTML de Confluence"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remover elementos no deseados
            for tag in soup(['script', 'style', 'meta', 'link']):
                tag.decompose()

            # Procesamiento especial para tablas
            for table in soup.find_all('table'):
                table_text = "\n=== TABLA ===\n"

                # Procesar encabezados
                headers = table.find_all('th')
                if headers:
                    header_row = " | ".join(
                        [h.get_text(strip=True) for h in headers])
                    table_text += f"COLUMNAS: {header_row}\n"

                # Procesar filas
                for row in table.find_all('tr'):
                    cells = row.find_all(['td', 'th'])
                    if cells:
                        cell_texts = [cell.get_text(strip=True)
                                      for cell in cells]
                        if any(cell_texts):
                            row_text = " | ".join(cell_texts)
                            table_text += f"FILA: {row_text}\n"

                table_text += "=== FIN TABLA ===\n"
                table.replace_with(table_text)

            # Convertir a texto
            text_content = self.html_converter.handle(str(soup))

            # Limpiezas adicionales
            text_content = re.sub(r'\n\s*\n\s*\n', '\n\n', text_content)
            text_content = re.sub(r' +', ' ', text_content)

            return text_content.strip()

        except Exception as e:
            logger.error(f"Error limpiando HTML: {e}")
            return self.html_converter.handle(html_content)

    def _categorize_document(self, title: str, content: str) -> str:
        """Categorizar documento según título y contenido"""
        title_lower = title.lower()

        if any(word in title_lower for word in ['pin', 'cvv', 'validacion']):
            return 'authentication'
        elif any(word in title_lower for word in ['bolsillos', 'wallet', 'saldo']):
            return 'wallet_management'
        elif any(word in title_lower for word in ['error', 'alarma', 'issue']):
            return 'error_handling'
        elif any(word in title_lower for word in ['transaccion', 'reversal', 'cambio']):
            return 'transaction_management'
        elif any(word in title_lower for word in ['tarjeta', 'blocked', 'bloqueada']):
            return 'card_management'
        elif any(word in title_lower for word in ['usuario', 'registrado']):
            return 'user_management'
        else:
            return 'technical_procedure'

    def _calculate_technical_score(self, content: str) -> float:
        """Calcular puntuación técnica del contenido"""
        score = 0.3  # Score base para documentos objetivo
        content_lower = content.lower()

        technical_indicators = [
            ('procedimiento', 0.2), ('paso', 0.15), ('error', 0.15),
            ('ticket', 0.1), ('api', 0.1), ('código', 0.1),
            ('escalamiento', 0.15), ('payment_processor', 0.1), ('issuing', 0.1)
        ]

        for indicator, weight in technical_indicators:
            if indicator in content_lower:
                score += weight

        # Bonificaciones adicionales
        if re.search(r'paso \d+', content_lower):
            score += 0.2
        if re.search(r'https?://', content_lower):
            score += 0.1

        return min(score, 1.0)

    # Métodos de compatibilidad con interface existente
    def extract_rada_folder_content(self, max_pages: int = None) -> List[Dict]:
        """Método principal - Extrae las páginas objetivo"""
        return self.extract_target_pages()

    def get_folder_content(self, folder_id: str = None) -> List[Dict]:
        """Alias para compatibilidad"""
        return self.extract_target_pages()

    def extract_all_space_content(self, max_pages: int = None) -> List[Dict]:
        """Alias para compatibilidad"""
        return self.extract_target_pages()

    def get_connection_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de extracción"""
        return {
            **self.stats,
            'success_rate': f"{(self.stats['pages_extracted'] / len(self.RADA_TARGET_PAGES) * 100):.1f}%",
            'target_pages': list(self.RADA_TARGET_PAGES.keys()),
            'extraction_method': 'direct_page_ids'
        }

    def format_for_rada(self, extracted_documents: List[Dict]) -> List[Dict]:
        """Formatear documentos para integración con RADA"""
        formatted_docs = []

        for doc in extracted_documents:
            content = doc['content']
            metadata = doc['metadata']

            rada_doc = {
                'text': content,
                'metadata': {
                    'source': f"Confluence RADA - {metadata['title']}",
                    'confluence_id': metadata['confluence_id'],
                    'title': metadata['title'],
                    'url': metadata['url'],
                    'space': metadata['space_name'],
                    'category': metadata['doc_category'],
                    'technical_score': metadata['technical_score'],
                    'created_by': metadata['created_by'],
                    'version': metadata['version'],
                    'extraction_date': metadata['extraction_date'],
                    'document_type': 'confluence_target',
                    'target_description': metadata['target_description'],
                    'extraction_method': metadata['extraction_method'],
                    'has_procedures': 'paso' in content.lower() or 'procedimiento' in content.lower(),
                    'has_api_info': 'api' in content.lower() or 'endpoint' in content.lower(),
                    'has_troubleshooting': any(word in content.lower() for word in ['error', 'problema', 'issue', 'alarma']),
                    'is_rada_content': True,
                    'curation_status': 'target_document'
                }
            }

            formatted_docs.append(rada_doc)

        return formatted_docs

    def get_target_pages_info(self) -> Dict[str, Any]:
        """Obtener información sobre las páginas objetivo"""
        return {
            'total_target_pages': len(self.RADA_TARGET_PAGES),
            'target_pages': self.RADA_TARGET_PAGES,
            'categories': {
                'authentication': ['12189000008'],
                'wallet_management': ['12015000006', '11842000010'],
                'error_handling': ['12021000009'],
                'transaction_management': ['11678000012', '12177000015', '12231000013'],
                'card_management': ['12207000011'],
                'user_management': ['12024000007'],
                'integrations': ['11467000014']
            },
            'extraction_method': 'direct_page_ids',
            'last_updated': datetime.now().isoformat()
        }
