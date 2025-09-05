"""
Motor de Cálculo Contextual para Consultas Comparativas
Calcula automaticamente crescimento, variações e comparações temporais/dimensionais
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import re
import numpy as np

class ComparativeCalculator:
    """Classe responsável por cálculos comparativos inteligentes"""
    
    def __init__(self):
        self.calculation_cache = {}
        self.supported_metrics = [
            'growth_rate', 'variation_percentage', 'absolute_change',
            'period_comparison', 'ranking_change', 'compound_growth'
        ]
    
    def detect_calculation_requirements(self, query: str, filters: dict) -> dict:
        """
        Detecta que tipo de cálculos são necessários baseado na query e filtros.
        
        Args:
            query: Query do usuário
            filters: Filtros expandidos
            
        Returns:
            dict: Requisitos de cálculo detectados
        """
        query_lower = query.lower()
        
        requirements = {
            'needs_growth_calculation': False,
            'needs_variation_calculation': False,
            'needs_period_comparison': False,
            'needs_ranking_comparison': False,
            'calculation_type': None,
            'metric_focus': None,
            'temporal_granularity': 'monthly',
            'comparison_baseline': None
        }
        
        # === DETECÇÃO DE TIPO DE CÁLCULO ===
        
        # Crescimento
        growth_keywords = ['cresceram', 'crescimento', 'crescer', 'aumentaram', 'subiu', 'subiram']
        if any(keyword in query_lower for keyword in growth_keywords):
            requirements['needs_growth_calculation'] = True
            requirements['calculation_type'] = 'growth'
        
        # Variação/Diferença
        variation_keywords = ['variação', 'diferença', 'mudança', 'alteração', 'variou']
        if any(keyword in query_lower for keyword in variation_keywords):
            requirements['needs_variation_calculation'] = True
            if not requirements['calculation_type']:
                requirements['calculation_type'] = 'variation'
        
        # Comparação de períodos
        comparison_keywords = ['entre', 'comparado', 'versus', 'vs', 'contra']
        if any(keyword in query_lower for keyword in comparison_keywords):
            requirements['needs_period_comparison'] = True
            if not requirements['calculation_type']:
                requirements['calculation_type'] = 'period_comparison'
        
        # Ranking/Posicionamento
        ranking_keywords = ['top', 'ranking', 'primeiro', 'último', 'melhor', 'pior', 'posição']
        if any(keyword in query_lower for keyword in ranking_keywords):
            requirements['needs_ranking_comparison'] = True
        
        # === DETECÇÃO DE MÉTRICA FOCO ===
        
        # Vendas/Receita
        if any(word in query_lower for word in ['vendas', 'receita', 'faturamento', 'valor']):
            requirements['metric_focus'] = 'revenue'
        
        # Volume/Quantidade
        elif any(word in query_lower for word in ['quantidade', 'volume', 'unidades', 'pçs']):
            requirements['metric_focus'] = 'quantity'
        
        # Clientes
        elif any(word in query_lower for word in ['clientes', 'cliente', 'compradores']):
            requirements['metric_focus'] = 'clients'
        
        # === DETECÇÃO DE GRANULARIDADE TEMPORAL ===
        
        if 'mês a mês' in query_lower or 'mensal' in query_lower:
            requirements['temporal_granularity'] = 'monthly'
        elif 'trimestre' in query_lower:
            requirements['temporal_granularity'] = 'quarterly'
        elif 'ano' in query_lower:
            requirements['temporal_granularity'] = 'yearly'
        
        # === DETECÇÃO DE BASELINE ===
        
        # Filtros indicam baseline temporal
        if '_comparative_period_analysis' in filters:
            requirements['comparison_baseline'] = 'temporal_periods'
        elif '_requires_growth_calculation' in filters:
            requirements['comparison_baseline'] = 'sequential_periods'
        
        return requirements
    
    def generate_comparative_sql_instructions(self, requirements: dict, filters: dict) -> str:
        """
        Gera instruções SQL específicas para cálculos comparativos.
        
        Args:
            requirements: Requisitos de cálculo detectados
            filters: Filtros expandidos
            
        Returns:
            str: Instruções SQL para cálculos comparativos
        """
        instructions = []
        
        # === CABEÇALHO BASEADO NO TIPO DE CÁLCULO ===
        if requirements['calculation_type'] == 'growth':
            instructions.append("ANÁLISE DE CRESCIMENTO AUTOMÁTICA:")
            instructions.append("Execute consulta que permita calcular crescimento entre períodos.")
        
        elif requirements['calculation_type'] == 'variation':
            instructions.append("ANÁLISE DE VARIAÇÃO AUTOMÁTICA:")
            instructions.append("Execute consulta que permita calcular variações percentuais e absolutas.")
        
        elif requirements['calculation_type'] == 'period_comparison':
            instructions.append("COMPARAÇÃO DE PERÍODOS AUTOMÁTICA:")
            instructions.append("Execute consulta separada para cada período e depois compare resultados.")
        
        # === INSTRUÇÕES ESPECÍFICAS PARA EXPANSÃO TEMPORAL ===
        
        if '_expand_temporal_analysis' in filters:
            instructions.append("\nEXPANSÃO TEMPORAL OBRIGATÓRIA:")
            instructions.append("- NÃO aplique filtro restritivo de data única")
            instructions.append("- INCLUA dados de TODOS os períodos relevantes para comparação")
            
            if '_temporal_range_start' in filters and '_temporal_range_end' in filters:
                start = filters['_temporal_range_start']
                end = filters['_temporal_range_end']
                instructions.append(f"- PERÍODO COMPLETO: de {start} até {end}")
        
        # === INSTRUÇÕES PARA CÁLCULOS DE CRESCIMENTO ===
        
        if requirements['needs_growth_calculation']:
            instructions.append("\nCÁLCULOS DE CRESCIMENTO NECESSÁRIOS:")
            instructions.append("- Agrupe dados por período (mês/trimestre/ano)")
            instructions.append("- Calcule métricas para cada período separadamente")
            instructions.append("- IMPORTANTE: Retorne dados estruturados para permitir cálculo de % crescimento")
            
            if requirements['metric_focus'] == 'revenue':
                instructions.append("- Foque em: valor total vendido, receita, faturamento")
            elif requirements['metric_focus'] == 'quantity':
                instructions.append("- Foque em: quantidade vendida, volume, unidades")
            elif requirements['metric_focus'] == 'clients':
                instructions.append("- Foque em: número de clientes, clientes únicos")
        
        # === INSTRUÇÕES PARA PRESERVAÇÃO DE CONTEXTO GEOGRÁFICO ===
        
        geographic_preserved = [key for key in filters.keys() if key.startswith('_preserve_')]
        if geographic_preserved:
            instructions.append("\nCONTEXTO GEOGRÁFICO PRESERVADO:")
            for key in geographic_preserved:
                field = key.replace('_preserve_', '')
                value = filters[key]
                instructions.append(f"- MANTER filtro: {field} = '{value}'")
        
        # === INSTRUÇÕES PARA SQL ESPECÍFICO ===
        
        instructions.append("\nESTRUTURA SQL RECOMENDADA:")
        
        if requirements['calculation_type'] == 'growth':
            instructions.append("SELECT")
            instructions.append("  DATE_TRUNC('month', Data) as periodo,")
            instructions.append("  SUM(valor_metrica) as total_periodo,")
            instructions.append("  COUNT(DISTINCT cliente) as clientes_periodo")
            instructions.append("FROM dados_comerciais")
            instructions.append("WHERE [aplicar_filtros_geograficos]")
            instructions.append("  AND Data BETWEEN '[inicio]' AND '[fim]'")
            instructions.append("GROUP BY DATE_TRUNC('month', Data)")
            instructions.append("ORDER BY periodo")
        
        elif requirements['needs_period_comparison']:
            instructions.append("-- Execute consulta com agrupamento por período")
            instructions.append("-- Estruture resultado para permitir comparação automática")
            
        # === INSTRUÇÕES FINAIS ===
        
        instructions.append("\nIMPORTANTE:")
        instructions.append("- NÃO pergunte confirmação ao usuário")
        instructions.append("- Execute a análise comparativa automaticamente")
        instructions.append("- Retorne dados estruturados com períodos claramente identificados")
        instructions.append("- Inclua cálculos percentuais quando apropriado")
        
        return "\n".join(instructions)
    
    def calculate_growth_metrics(self, period_data: pd.DataFrame, metric_column: str) -> Dict[str, Any]:
        """
        Calcula métricas de crescimento a partir de dados por período.
        
        Args:
            period_data: DataFrame com dados agrupados por período
            metric_column: Nome da coluna com a métrica principal
            
        Returns:
            dict: Métricas de crescimento calculadas
        """
        if len(period_data) < 2:
            return {"error": "Necessários pelo menos 2 períodos para calcular crescimento"}
        
        # Ordenar por período
        period_data = period_data.sort_values('periodo')
        
        growth_metrics = {
            'period_count': len(period_data),
            'total_growth': None,
            'period_growth_rates': [],
            'average_growth_rate': None,
            'best_period': None,
            'worst_period': None,
            'trend': None
        }
        
        # === CÁLCULO DE CRESCIMENTO PERÍODO A PERÍODO ===
        
        values = period_data[metric_column].values
        periods = period_data['periodo'].values
        
        growth_rates = []
        
        for i in range(1, len(values)):
            if values[i-1] != 0:  # Evitar divisão por zero
                growth_rate = ((values[i] - values[i-1]) / values[i-1]) * 100
                growth_rates.append({
                    'from_period': periods[i-1],
                    'to_period': periods[i],
                    'growth_rate': growth_rate,
                    'absolute_change': values[i] - values[i-1]
                })
        
        growth_metrics['period_growth_rates'] = growth_rates
        
        # === MÉTRICAS AGREGADAS ===
        
        if growth_rates:
            # Crescimento total (primeira vs última)
            if values[0] != 0:
                total_growth = ((values[-1] - values[0]) / values[0]) * 100
                growth_metrics['total_growth'] = total_growth
            
            # Taxa média de crescimento
            avg_growth = np.mean([gr['growth_rate'] for gr in growth_rates])
            growth_metrics['average_growth_rate'] = avg_growth
            
            # Melhor e pior período
            best_growth = max(growth_rates, key=lambda x: x['growth_rate'])
            worst_growth = min(growth_rates, key=lambda x: x['growth_rate'])
            
            growth_metrics['best_period'] = best_growth
            growth_metrics['worst_period'] = worst_growth
            
            # Detecção de tendência
            if len(growth_rates) >= 3:
                recent_rates = [gr['growth_rate'] for gr in growth_rates[-3:]]
                if all(recent_rates[i] > recent_rates[i-1] for i in range(1, len(recent_rates))):
                    growth_metrics['trend'] = 'accelerating'
                elif all(recent_rates[i] < recent_rates[i-1] for i in range(1, len(recent_rates))):
                    growth_metrics['trend'] = 'decelerating'
                else:
                    growth_metrics['trend'] = 'variable'
        
        return growth_metrics
    
    def generate_comparative_summary(self, growth_metrics: Dict[str, Any], requirements: dict) -> str:
        """
        Gera um resumo textual dos resultados comparativos.
        
        Args:
            growth_metrics: Métricas calculadas
            requirements: Requisitos originais
            
        Returns:
            str: Resumo formatado dos resultados
        """
        if 'error' in growth_metrics:
            return f"Erro no cálculo: {growth_metrics['error']}"
        
        summary_parts = []
        
        # === CABEÇALHO BASEADO NO TIPO ===
        if requirements['calculation_type'] == 'growth':
            summary_parts.append("## ANÁLISE DE CRESCIMENTO AUTOMÁTICA")
        else:
            summary_parts.append("## ANÁLISE COMPARATIVA AUTOMÁTICA")
        
        # === RESUMO DOS NÚMEROS PRINCIPAIS ===
        
        if growth_metrics.get('total_growth'):
            total_growth = growth_metrics['total_growth']
            summary_parts.append(f"\n**Crescimento Total**: {total_growth:.1f}%")
        
        if growth_metrics.get('average_growth_rate'):
            avg_growth = growth_metrics['average_growth_rate']
            summary_parts.append(f"**Taxa Média**: {avg_growth:.1f}% por período")
        
        # === DESTAQUES POR PERÍODO ===
        
        if growth_metrics.get('best_period'):
            best = growth_metrics['best_period']
            summary_parts.append(f"\n**Melhor Performance**: {best['growth_rate']:.1f}% ({best['from_period']} → {best['to_period']})")
        
        if growth_metrics.get('worst_period'):
            worst = growth_metrics['worst_period']
            summary_parts.append(f"**Pior Performance**: {worst['growth_rate']:.1f}% ({worst['from_period']} → {worst['to_period']})")
        
        # === ANÁLISE DE TENDÊNCIA ===
        
        if growth_metrics.get('trend'):
            trend = growth_metrics['trend']
            if trend == 'accelerating':
                summary_parts.append(f"\n**Tendência**: 📈 Crescimento acelerando")
            elif trend == 'decelerating':
                summary_parts.append(f"\n**Tendência**: 📉 Crescimento desacelerando")
            else:
                summary_parts.append(f"\n**Tendência**: 📊 Performance variável")
        
        # === DETALHAMENTO PERÍODO A PERÍODO ===
        
        if growth_metrics.get('period_growth_rates'):
            summary_parts.append("\n### Evolução Período a Período")
            
            for growth in growth_metrics['period_growth_rates']:
                from_p = growth['from_period']
                to_p = growth['to_period']
                rate = growth['growth_rate']
                
                emoji = "📈" if rate > 0 else "📉" if rate < 0 else "➡️"
                summary_parts.append(f"- {from_p} → {to_p}: {rate:+.1f}% {emoji}")
        
        return "\n".join(summary_parts)

# Instância global para uso conveniente
comparative_calculator = ComparativeCalculator()