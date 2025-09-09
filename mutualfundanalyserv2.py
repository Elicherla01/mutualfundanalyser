import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import re

def safe_display_value(value):
    """Safely convert any value to a displayable format"""
    if pd.isna(value) or value is None:
        return "N/A"
    elif hasattr(value, 'strftime'):  # Timestamp object
        return value.strftime('%Y-%m-%d')
    elif isinstance(value, (int, float)):
        return value
    else:
        return str(value)

# Page configuration
st.set_page_config(
    page_title="Fund Performance Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Mutual Fund Performance Analysis Dashboard")

# Made in India section
col1, col2 = st.columns([2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 8px; background: linear-gradient(90deg, rgba(255, 153, 51, 0.3), rgba(255, 255, 255, 0.3), rgba(19, 136, 8, 0.3)); border-radius: 8px; margin: 10px 0;">
        <h5 style="color: #000080; margin: 0; font-weight: bold;">
            🇮🇳 Made in India for India 🇮🇳
        </h5>
    </div>
    """, unsafe_allow_html=True)

# Social sharing and action buttons
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col1:
    st.markdown("""
    <div style="text-align: center; padding: 5px;">
        <span style="color: #666; font-size: 14px;">Share</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 5px;">
        <span style="color: #666; font-size: 18px;">⭐</span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 5px;">
        <span style="color: #666; font-size: 18px;">✏️</span>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="text-align: center; padding: 5px;">
        <a href="https://github.com/Elicherla01/mutualfundanalyser" target="_blank" style="text-decoration: none;">
            <span style="color: #666; font-size: 18px;">🐙</span>
        </a>
    </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown("""
    <div style="text-align: center; padding: 5px;">
        <span style="color: #666; font-size: 18px;">⋯</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# File upload
uploaded_file = st.file_uploader(
    "Upload your Fund Performance Excel file", 
    type=['xlsx', 'xls'],
    help="Upload the FundPerformance Excel file to begin analysis"
)

if uploaded_file is not None:
    try:
        # Read the Excel file
        df = pd.read_excel(uploaded_file, header=4)  # Headers are in row 5 (index 4)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Filter out empty rows and clean data
        df = df.dropna(subset=['Scheme Name'])
        df = df[df['Scheme Name'].str.strip() != '']
        
        # Additional filtering to remove rows with invalid scheme names
        df = df[~df['Scheme Name'].str.contains(r'^[0-9\s\-_]+$', na=False)]  # Remove rows with only numbers/special chars
        df = df[df['Scheme Name'].str.len() > 2]  # Remove very short names
        
        # Additional strict filtering to ensure only valid fund names
        df = df[df['Scheme Name'].str.len() > 5]  # Remove very short names (more strict)
        df = df[~df['Scheme Name'].str.contains(r'^[A-Z\s]{1,10}$', na=False)]  # Remove very short all-caps entries
        df = df[~df['Scheme Name'].str.contains(r'[0-9]{4,}', na=False)]  # Remove entries with 4+ consecutive digits
        df = df[df['Scheme Name'].str.contains(r'[a-zA-Z]', na=False)]  # Must contain at least one letter
        
        # Remove specific non-fund entries
        df = df[~df['Scheme Name'].str.contains(r'\*For detailed understanding regarding Information Ratio, click on the below link', na=False, case=False)]
        df = df[~df['Scheme Name'].str.contains(r'amfiindia\.com/information-ratio', na=False, case=False)]
        
        # Clean the dataframe to handle any problematic data types
        for col in df.columns:
            try:
                # Try to convert to appropriate type, fallback to string if issues
                if df[col].dtype == 'object':
                    # Check if it's numeric data
                    if df[col].str.match(r'^-?\d*\.?\d+$').all():
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    else:
                        # Convert to string and handle any remaining issues
                        df[col] = df[col].astype(str)
            except:
                # If any conversion fails, keep as string
                df[col] = df[col].astype(str)
        
        # Final count after all filtering
        final_count = len(df)
        final_count = final_count
        st.success(f"✅ Data loaded successfully! Found {final_count} valid funds after filtering")
        
        # Show filtering details
        with st.expander("Data Filtering Details"):
            original_df = pd.read_excel(uploaded_file, header=4)
            st.write(f"**Original rows in file:** {len(original_df)}")
            
            # Show what was filtered out
            st.write("**Rows that were filtered out:**")
            filtered_out = []
            for idx, row in original_df.iterrows():
                scheme_name = str(row.get('Scheme Name', '')).strip()
                if pd.isna(scheme_name) or scheme_name == '' or scheme_name == 'nan':
                    filtered_out.append(f"Row {idx+5}: Empty/NaN scheme name")
                elif re.match(r'^[0-9\s\-_]+$', scheme_name):
                    filtered_out.append(f"Row {idx+5}: '{scheme_name}' - Only numbers/special characters")
                elif len(scheme_name) <= 2:
                    filtered_out.append(f"Row {idx+5}: '{scheme_name}' - Too short")
            
            if filtered_out:
                for item in filtered_out:
                    st.write(f"- {item}")
            else:
                st.write("No rows were filtered out")
            
            st.write(f"**Final valid funds:** {final_count}")
            st.write("**Filtering criteria:**")
            st.write("- Removed rows with empty or missing scheme names")
            st.write("- Removed rows with only numbers/special characters")
            st.write("- Removed rows with very short names (< 3 characters)")
            st.write("- Removed non-fund entries (Information Ratio links and AMFI website references)")
            
            # Add detailed debugging for the final filtered data
            st.write("**Detailed Analysis of Final Data:**")
            st.write("**All final fund names with row numbers:**")
            for idx, row in df.iterrows():
                original_row_num = idx + 5  # Convert back to Excel row number
                scheme_name = row['Scheme Name']
                st.write(f"Row {original_row_num}: '{scheme_name}'")
            
            # Check for any suspicious entries
            st.write("**Potential Issues Check:**")
            suspicious_funds = []
            for idx, row in df.iterrows():
                scheme_name = str(row['Scheme Name']).strip()
                if len(scheme_name) <= 5:  # Very short names
                    suspicious_funds.append(f"Row {idx+5}: '{scheme_name}' (very short)")
                elif re.search(r'[0-9]{4,}', scheme_name):  # Contains 4+ consecutive digits
                    suspicious_funds.append(f"Row {idx+5}: '{scheme_name}' (contains many digits)")
                elif re.search(r'^[A-Z\s]+$', scheme_name) and len(scheme_name) <= 10:  # All caps and short
                    suspicious_funds.append(f"Row {idx+5}: '{scheme_name}' (all caps and short)")
            
            if suspicious_funds:
                st.write("**Potentially problematic entries:**")
                for item in suspicious_funds:
                    st.write(f"- {item}")
            else:
                st.write("No suspicious entries found")
        
        # Display basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Funds", final_count)
        with col2:
            benchmark = df['Benchmark'].iloc[0] if not df['Benchmark'].empty else "N/A"
            benchmark = safe_display_value(benchmark)
            st.metric("Common Benchmark", benchmark)
        with col3:
            nav_date = df['NAV Date'].iloc[0] if not df['NAV Date'].empty else "N/A"
            # Convert timestamp to string if it's a timestamp object
            if hasattr(nav_date, 'strftime'):
                nav_date = nav_date.strftime('%Y-%m-%d')
            st.metric("NAV Date", nav_date)
        
        # Show list of valid fund names
        with st.expander("List of Valid Fund Names"):
            st.write(f"**Total valid funds: {final_count}**")
            for i, fund_name in enumerate(df['Scheme Name'].sort_values(), 1):
                st.write(f"{i}. {fund_name}")
        
        # Sidebar for analysis options
        st.sidebar.header("Analysis Options")
        
        analysis_type = st.sidebar.selectbox(
            "Choose Analysis Type",
            ["Performance Overview", "Timeline Comparison", "Consistency Analysis", "AUM Correlation", "Risk-Return Analysis", "Performance vs Benchmark Heatmap", "Investment Calculator"]
        )
        
        # Convert return columns to numeric
        return_columns = [
            'Return 1 Year (%) Regular', 'Return 1 Year (%) Direct',
            'Return 3 Year (%) Regular', 'Return 3 Year (%) Direct', 
            'Return 5 Year (%) Regular', 'Return 5 Year (%) Direct',
            'Return 10 Year (%) Regular', 'Return 10 Year (%) Direct',
            'Return 1 Year (%) Benchmark', 'Return 3 Year (%) Benchmark',
            'Return 5 Year (%) Benchmark', 'Return 10 Year (%) Benchmark'
        ]
        
        for col in return_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    st.warning(f"Could not convert column '{col}' to numeric: {str(e)}")
                    df[col] = df[col].astype(str)
        
        # Convert AUM to numeric
        if 'Daily AUM (Cr.)' in df.columns:
            try:
                df['Daily AUM (Cr.)'] = pd.to_numeric(df['Daily AUM (Cr.)'], errors='coerce')
            except Exception as e:
                st.warning(f"Could not convert AUM column to numeric: {str(e)}")
                df['Daily AUM (Cr.)'] = df['Daily AUM (Cr.)'].astype(str)
        
        # Handle timestamp columns - convert to string if needed
        timestamp_columns = ['NAV Date', 'Launch Date', 'Inception Date']
        for col in timestamp_columns:
            if col in df.columns:
                try:
                    df[col] = df[col].astype(str)
                except Exception as e:
                    st.warning(f"Could not convert timestamp column '{col}': {str(e)}")
                    # Try to handle as datetime first, then convert to string
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
                    except:
                        df[col] = df[col].astype(str)
        
        # Define time periods globally for all analysis sections
        time_periods = ['1 Year', '3 Year', '5 Year', '10 Year']
        
        # Analysis sections
        if analysis_type == "Performance Overview":
            st.header("📊 Performance Overview")
            
            # Performance summary table
            st.subheader("Performance Summary Across Time Periods")
            
            # Create summary statistics
            summary_data = []
            
            for period in time_periods:
                regular_col = f'Return {period} (%) Regular'
                direct_col = f'Return {period} (%) Direct'
                benchmark_col = f'Return {period} (%) Benchmark'
                
                if regular_col in df.columns and direct_col in df.columns:
                    regular_data = df[regular_col].dropna()
                    direct_data = df[direct_col].dropna()
                    benchmark_data = df[benchmark_col].dropna()
                    
                    summary_data.append({
                        'Time Period': period,
                        'Funds with Data (Regular)': len(regular_data),
                        'Avg Return Regular (%)': regular_data.mean(),
                        'Funds with Data (Direct)': len(direct_data),
                        'Avg Return Direct (%)': direct_data.mean(),
                        'Avg Benchmark Return (%)': benchmark_data.mean(),
                        'Avg Outperformance Regular (%)': (regular_data - benchmark_data.iloc[0]).mean() if len(benchmark_data) > 0 else np.nan,
                        'Avg Outperformance Direct (%)': (direct_data - benchmark_data.iloc[0]).mean() if len(benchmark_data) > 0 else np.nan
                    })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df.round(2), use_container_width=True)
            
            # Performance distribution charts
            st.subheader("Return Distribution by Time Period")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Regular plan returns - Bubble Chart
                fig_reg = go.Figure()
                for period in time_periods:
                    col_name = f'Return {period} (%) Regular'
                    if col_name in df.columns:
                        data = df[col_name].dropna()
                        if len(data) > 0:
                            # Create bubble chart with period on x-axis and returns on y-axis
                            fig_reg.add_trace(go.Scatter(
                                x=[period] * len(data),
                                y=data,
                                mode='markers',
                                name=period,
                                marker=dict(
                                    size=8,
                                    opacity=0.7,
                                    color='lightblue'
                                ),
                                text=df.loc[data.index, 'Scheme Name'],
                                hovertemplate='<b>%{text}</b><br>' +
                                            f'{period} Return: %{{y:.2f}}%<br>' +
                                            '<extra></extra>'
                            ))
                
                fig_reg.update_layout(
                    title="Regular Plan Returns Distribution (Bubble Chart)",
                    xaxis_title="Time Period",
                    yaxis_title="Returns (%)",
                    showlegend=False,
                    hovermode='closest'
                )
                fig_reg.update_xaxes(categoryorder='array', categoryarray=time_periods)
                st.plotly_chart(fig_reg, use_container_width=True)
            
            with col2:
                # Direct plan returns - Bubble Chart
                fig_dir = go.Figure()
                for period in time_periods:
                    col_name = f'Return {period} (%) Direct'
                    if col_name in df.columns:
                        data = df[col_name].dropna()
                        if len(data) > 0:
                            # Create bubble chart with period on x-axis and returns on y-axis
                            fig_dir.add_trace(go.Scatter(
                                x=[period] * len(data),
                                y=data,
                                mode='markers',
                                name=period,
                                marker=dict(
                                    size=8,
                                    opacity=0.7,
                                    color='lightcoral'
                                ),
                                text=df.loc[data.index, 'Scheme Name'],
                                hovertemplate='<b>%{text}</b><br>' +
                                            f'{period} Return: %{{y:.2f}}%<br>' +
                                            '<extra></extra>'
                            ))
                
                fig_dir.update_layout(
                    title="Direct Plan Returns Distribution (Bubble Chart)",
                    xaxis_title="Time Period",
                    yaxis_title="Returns (%)",
                    showlegend=False,
                    hovermode='closest'
                )
                fig_dir.update_xaxes(categoryorder='array', categoryarray=time_periods)
                st.plotly_chart(fig_dir, use_container_width=True)
            
            # Combined scatter plot for all time periods
            st.subheader("Combined View: All Time Periods Comparison")
            
            # Create a combined scatter plot
            fig_combined = go.Figure()
            
            colors = ['lightblue', 'darkblue', 'lightcoral', 'red']
            
            for i, period in enumerate(time_periods):
                regular_col = f'Return {period} (%) Regular'
                direct_col = f'Return {period} (%) Direct'
                
                if regular_col in df.columns:
                    regular_data = df[regular_col].dropna()
                    if len(regular_data) > 0:
                        fig_combined.add_trace(go.Scatter(
                            x=[period] * len(regular_data),
                            y=regular_data,
                            mode='markers',
                            name=f'{period} Regular',
                            marker=dict(
                                size=8,
                                opacity=0.7,
                                color=colors[i],
                                symbol='circle'
                            ),
                            text=df.loc[regular_data.index, 'Scheme Name'],
                            hovertemplate='<b>%{text}</b><br>' +
                                        f'{period} Regular: %{{y:.2f}}%<br>' +
                                        '<extra></extra>'
                        ))
                
                if direct_col in df.columns:
                    direct_data = df[direct_col].dropna()
                    if len(direct_data) > 0:
                        fig_combined.add_trace(go.Scatter(
                            x=[period] * len(direct_data),
                            y=direct_data,
                            mode='markers',
                            name=f'{period} Direct',
                            marker=dict(
                                size=8,
                                opacity=0.7,
                                color=colors[i],
                                symbol='diamond'
                            ),
                            text=df.loc[direct_data.index, 'Scheme Name'],
                            hovertemplate='<b>%{text}</b><br>' +
                                        f'{period} Direct: %{{y:.2f}}%<br>' +
                                        '<extra></extra>'
                        ))
            
            fig_combined.update_layout(
                title="All Time Periods - Regular vs Direct Returns Comparison",
                xaxis_title="Time Period",
                yaxis_title="Returns (%)",
                showlegend=True,
                hovermode='closest',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            fig_combined.update_xaxes(categoryorder='array', categoryarray=time_periods)
            st.plotly_chart(fig_combined, use_container_width=True)
        
        elif analysis_type == "Timeline Comparison":
            st.header("⏱️ Timeline Comparison Analysis")
            
            # Top performers across different timelines
            st.subheader("Top 10 Performers by Timeline")
            
            tabs = st.tabs(time_periods)
            
            for i, (tab, period) in enumerate(zip(tabs, time_periods)):
                with tab:
                    direct_col = f'Return {period} (%) Direct'
                    regular_col = f'Return {period} (%) Regular'
                    benchmark_col = f'Return {period} (%) Benchmark'
                    
                    if direct_col in df.columns:
                        # Get benchmark return for this period
                        benchmark_return = None
                        if benchmark_col in df.columns:
                            # Find the first non-NaN benchmark value
                            benchmark_data = df[benchmark_col].dropna()
                            if len(benchmark_data) > 0:
                                benchmark_return = benchmark_data.iloc[0]
                            else:
                                st.warning(f"No benchmark data available for {period}")
                        else:
                            st.warning(f"Benchmark column '{benchmark_col}' not found in data")
                        
                        # Create comparison dataframe
                        comparison_df = df[['Scheme Name', direct_col, regular_col, 'Daily AUM (Cr.)', benchmark_col]].copy()
                        comparison_df = comparison_df.dropna(subset=[direct_col])
                        comparison_df = comparison_df.sort_values(direct_col, ascending=False)
                        
                        # Add rank column based on Direct Fund returns
                        comparison_df['Rank'] = range(1, len(comparison_df) + 1)
                        
                        # Add benchmark comparison columns using individual fund benchmarks
                        if benchmark_col in comparison_df.columns:
                            # Calculate outperformance using each fund's individual benchmark
                            comparison_df['Direct vs Benchmark (%)'] = comparison_df[direct_col] - comparison_df[benchmark_col]
                            comparison_df['Regular vs Benchmark (%)'] = comparison_df[regular_col] - comparison_df[benchmark_col]
                            
                            # Show benchmark data info
                            st.info(f"📊 Using individual fund benchmarks for {period} - each fund compared against its own benchmark")
                            
                            # Color code the outperformance with three distinct ranges
                            def color_outperformance(val):
                                if val > 4:
                                    return 'background-color: #2E8B57; color: white'  # Dark green for >4%
                                elif val > 0:
                                    return 'background-color: #FF8C00; color: white'  # Dark orange for 0-4%
                                elif val < 0:
                                    return 'background-color: #8B0000; color: white'  # Dark red for <0%
                                else:
                                    return 'background-color: #696969; color: white'  # Dark gray for exactly 0%
                            
                            # Display table with styling
                            st.write(f"**Individual Fund Benchmarks for {period}**")
                            # Round the data first, then apply styling
                            comparison_df_rounded = comparison_df.round(2)
                            # Reorder columns to show Rank first, then individual benchmark
                            comparison_df_rounded = comparison_df_rounded[['Rank', 'Scheme Name', direct_col, regular_col, benchmark_col, 'Daily AUM (Cr.)', 'Direct vs Benchmark (%)', 'Regular vs Benchmark (%)']]
                            styled_df = comparison_df_rounded.style.applymap(color_outperformance, subset=['Direct vs Benchmark (%)', 'Regular vs Benchmark (%)'])
                            st.dataframe(styled_df, use_container_width=True)
                        else:
                            # Debug: Show what benchmark data is available
                            if benchmark_col in df.columns:
                                benchmark_values = df[benchmark_col].dropna().unique()
                                st.warning(f"⚠️ Benchmark column exists but no valid data found. Available values: {benchmark_values}")
                            else:
                                st.error(f"❌ Benchmark column '{benchmark_col}' not found in dataset")
                            
                            # Display table without benchmark if not available
                            # Reorder columns to show Rank first
                            comparison_df_display = comparison_df.round(2)[['Rank', 'Scheme Name', direct_col, regular_col, benchmark_col, 'Daily AUM (Cr.)']]
                            st.dataframe(comparison_df_display, use_container_width=True)
                        
                        # Bar chart - show all funds ranked by performance
                        fig = px.bar(
                            comparison_df, 
                            x='Scheme Name', 
                            y=[direct_col, regular_col],
                            title=f"All Funds Ranked by {period} Returns (Sorted by Direct Plan Performance)",
                            barmode='group'
                        )
                        fig.update_layout(xaxis_tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
            
            # Regular vs Direct comparison
            st.subheader("Regular vs Direct Plan Performance")
            
            selected_period = st.selectbox("Select Time Period for Detailed Comparison", time_periods)
            
            regular_col = f'Return {selected_period} (%) Regular'
            direct_col = f'Return {selected_period} (%) Direct'
            
            if regular_col in df.columns and direct_col in df.columns:
                # Scatter plot with fund names and AUM-based bubble sizes
                valid_data = df[[regular_col, direct_col, 'Scheme Name', 'Daily AUM (Cr.)']].dropna()
                
                # Normalize AUM for bubble sizes (scale between 10 and 50)
                if len(valid_data) > 0:
                    min_aum = valid_data['Daily AUM (Cr.)'].min()
                    max_aum = valid_data['Daily AUM (Cr.)'].max()
                    if max_aum > min_aum:
                        valid_data = valid_data.copy()
                        valid_data['Bubble Size'] = 10 + (valid_data['Daily AUM (Cr.)'] - min_aum) / (max_aum - min_aum) * 40
                    else:
                        valid_data = valid_data.copy()
                        valid_data['Bubble Size'] = 30  # Default size if all AUM values are same
                else:
                    valid_data = valid_data.copy()
                    valid_data['Bubble Size'] = 30
                
                fig = px.scatter(
                    valid_data,
                    x=regular_col,
                    y=direct_col,
                    size='Bubble Size',
                    hover_data=['Scheme Name', 'Daily AUM (Cr.)'],
                    title=f"Regular vs Direct Returns - {selected_period}<br><sub>Bubble size represents AUM (in Crores)</sub>",
                    labels={regular_col: "Regular Plan Returns (%)", direct_col: "Direct Plan Returns (%)"},
                    color='Daily AUM (Cr.)',
                    color_continuous_scale='Viridis'
                )
                
                # Add fund names as text annotations
                for i, row in valid_data.iterrows():
                    fig.add_annotation(
                        x=row[regular_col],
                        y=row[direct_col],
                        text=row['Scheme Name'][:20] + "..." if len(row['Scheme Name']) > 20 else row['Scheme Name'],
                        showarrow=False,
                        font=dict(size=8, color="white"),
                        bgcolor="rgba(0,0,0,0.7)",
                        bordercolor="rgba(255,255,255,0.2)",
                        borderwidth=1,
                        yshift=15
                    )
                
                # Add diagonal line (where regular = direct)
                min_val = min(valid_data[regular_col].min(), valid_data[direct_col].min())
                max_val = max(valid_data[regular_col].max(), valid_data[direct_col].max())
                fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, 
                             line=dict(dash="dash", color="red", width=2))
                
                # Apply dark theme styling
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    xaxis=dict(
                        gridcolor='rgba(128,128,128,0.3)',
                        linecolor='rgba(128,128,128,0.5)'
                    ),
                    yaxis=dict(
                        gridcolor='rgba(128,128,128,0.3)',
                        linecolor='rgba(128,128,128,0.5)'
                    ),
                    coloraxis_colorbar=dict(
                        title="AUM (Cr.)",
                        title_font=dict(color='white'),
                        tickfont=dict(color='white')
                    ),
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Direct plan advantage
                valid_data['Direct Advantage'] = valid_data[direct_col] - valid_data[regular_col]
                avg_advantage = valid_data['Direct Advantage'].mean()
                st.metric(f"Average Direct Plan Advantage - {selected_period}", f"{avg_advantage:.2f}%")
        
        elif analysis_type == "Consistency Analysis":
            st.header("🎯 Performance Consistency Analysis")
            
            # Add dropdown for minimum years filter
            st.subheader("Analysis Configuration")
            
            # Show data availability summary first
            st.write("**Data Availability Summary:**")
            data_availability = []
            for period in time_periods:
                direct_col = f'Return {period} (%) Direct'
                if direct_col in df.columns:
                    available_funds = len(df[direct_col].dropna())
                    data_availability.append(f"{period}: {available_funds} funds")
            
            col1, col2, col3, col4 = st.columns(4)
            for i, availability in enumerate(data_availability):
                with [col1, col2, col3, col4][i]:
                    st.metric(availability.split(':')[0], availability.split(':')[1])
            
            # Debug: Show sample of fund data availability
            with st.expander("🔍 Debug: Sample Fund Data Availability"):
                st.write("**Sample of first 10 funds and their data availability:**")
                debug_data = []
                for idx, row in df.head(10).iterrows():
                    fund_name = row['Scheme Name']
                    available_periods = []
                    
                    for period in time_periods:
                        direct_col = f'Return {period} (%) Direct'
                        if direct_col in df.columns and not pd.isna(row[direct_col]):
                            available_periods.append(period)
                    
                    years_of_existence = 0
                    if available_periods:
                        years_of_existence = max([int(p.split()[0]) for p in available_periods])
                    
                    debug_data.append({
                        'Fund Name': fund_name,
                        'Available Periods': ', '.join(available_periods) if available_periods else 'None',
                        'Years of Existence': years_of_existence
                    })
                
                debug_df = pd.DataFrame(debug_data)
                st.dataframe(debug_df, use_container_width=True)
            
            min_years_filter = st.selectbox(
                "Select Minimum Years of Data Required",
                ["1 Year", "3 Years", "5 Years", "10 Years"],
                index=1,  # Default to 3 Years
                help="Only funds with data for at least this many years will be included in the consistency analysis"
            )
            
            # Convert selection to number
            min_years_map = {"1 Year": 1, "3 Years": 3, "5 Years": 5, "10 Years": 10}
            min_years_required = min_years_map[min_years_filter]
            
            # Calculate consistency metrics
            st.subheader("Outperformance Consistency")
            
            # Get benchmark returns - find the most common benchmark value for each period
            benchmark_returns = {}
            for period in time_periods:
                benchmark_col = f'Return {period} (%) Benchmark'
                if benchmark_col in df.columns:
                    # Get all non-NaN benchmark values for this period
                    benchmark_data = df[benchmark_col].dropna()
                    if len(benchmark_data) > 0:
                        # Use the most common benchmark value (mode) or the first non-NaN value
                        benchmark_values = benchmark_data.unique()
                        if len(benchmark_values) == 1:
                            # All funds have the same benchmark
                            benchmark_returns[period] = benchmark_values[0]
                        else:
                            # Multiple benchmark values - use the most frequent one
                            benchmark_counts = benchmark_data.value_counts()
                            benchmark_returns[period] = benchmark_counts.index[0]
                            st.warning(f"⚠️ Multiple benchmark values found for {period}: {benchmark_values}. Using most frequent: {benchmark_returns[period]}")
                    else:
                        benchmark_returns[period] = None
                        st.warning(f"⚠️ No benchmark data available for {period}")
                else:
                    benchmark_returns[period] = None
                    st.warning(f"⚠️ Benchmark column '{benchmark_col}' not found")
            
            # Debug: Show benchmark data
            st.info("📊 **Benchmark Data Used for Consistency Analysis:**")
            for period, benchmark in benchmark_returns.items():
                if benchmark is not None:
                    st.write(f"**{period}**: {benchmark:.2f}%")
                else:
                    st.write(f"**{period}**: No data")
            
            # Calculate outperformance for each fund across periods
            consistency_data = []
            
            for idx, row in df.iterrows():
                fund_data = {'Fund Name': row['Scheme Name'], 'AUM (Rs. Crores)': row.get('Daily AUM (Cr.)', np.nan)}
                outperform_count_direct = 0
                outperform_count_regular = 0
                total_periods_direct = 0
                total_periods_regular = 0
                
                for period in time_periods:
                    direct_col = f'Return {period} (%) Direct'
                    regular_col = f'Return {period} (%) Regular'
                    
                    if direct_col in df.columns and not pd.isna(row[direct_col]):
                        if benchmark_returns.get(period) is not None:
                            direct_outperform = row[direct_col] - benchmark_returns[period]
                            fund_data[f'{period} Direct Outperform'] = direct_outperform
                            if direct_outperform > 0:
                                outperform_count_direct += 1
                            total_periods_direct += 1
                    
                    if regular_col in df.columns and not pd.isna(row[regular_col]):
                        if benchmark_returns.get(period) is not None:
                            regular_outperform = row[regular_col] - benchmark_returns[period]
                            fund_data[f'{period} Regular Outperform'] = regular_outperform
                            if regular_outperform > 0:
                                outperform_count_regular += 1
                            total_periods_regular += 1
                
                # Calculate consistency ratios
                fund_data['Direct Consistency Ratio'] = outperform_count_direct / total_periods_direct if total_periods_direct > 0 else np.nan
                fund_data['Regular Consistency Ratio'] = outperform_count_regular / total_periods_regular if total_periods_regular > 0 else np.nan
                
                # Calculate years of existence based on available data
                years_of_existence = 0
                available_periods = []
                
                # Check which periods have data
                if 'Return 1 Year (%) Direct' in df.columns and not pd.isna(row['Return 1 Year (%) Direct']):
                    available_periods.append(1)
                if 'Return 3 Year (%) Direct' in df.columns and not pd.isna(row['Return 3 Year (%) Direct']):
                    available_periods.append(3)
                if 'Return 5 Year (%) Direct' in df.columns and not pd.isna(row['Return 5 Year (%) Direct']):
                    available_periods.append(5)
                if 'Return 10 Year (%) Direct' in df.columns and not pd.isna(row['Return 10 Year (%) Direct']):
                    available_periods.append(10)
                
                # Years of existence is the maximum period with data
                years_of_existence = max(available_periods) if available_periods else 0
                
                fund_data['Years of Existence'] = years_of_existence
                
                # Only include funds that have data for at least the selected minimum years
                if years_of_existence >= min_years_required:
                    consistency_data.append(fund_data)
            
            consistency_df = pd.DataFrame(consistency_data)
            
            # Show filtering summary
            total_funds = len(df)
            included_funds = len(consistency_df)
            excluded_funds = total_funds - included_funds
            
            st.info(f"📊 **Data Quality Filter Applied**: Only funds with data for at least {min_years_required} year{'s' if min_years_required > 1 else ''} are included in consistency analysis.")
            st.info(f"✅ **Funds Included**: {included_funds} out of {total_funds} total funds (≥{min_years_required} year{'s' if min_years_required > 1 else ''} of data)")
            if excluded_funds > 0:
                st.warning(f"⚠️ **Funds Excluded**: {excluded_funds} funds excluded due to insufficient data (<{min_years_required} year{'s' if min_years_required > 1 else ''})")
            
            # If no funds meet the criteria, show helpful message and skip the rest
            if included_funds == 0:
                st.error(f"❌ **No Analysis Possible**: No funds have data for at least {min_years_required} year{'s' if min_years_required > 1 else ''}. Please try selecting a lower minimum years requirement.")
                st.info("💡 **Suggestion**: Try selecting '1 Year' or '3 Years' to include more funds in the analysis.")
            else:
                # Display consistency rankings
                st.subheader("Most Consistent Outperformers (Direct Plans)")
                
                # Check if we have any data and if the required columns exist
                if len(consistency_df) > 0 and 'Direct Consistency Ratio' in consistency_df.columns:
                    top_consistent = consistency_df.dropna(subset=['Direct Consistency Ratio']).sort_values(
                        'Direct Consistency Ratio', ascending=False
                    ).head(10)
                    
                    if len(top_consistent) > 0:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Check which columns are available for display
                            available_columns = []
                            for col in ['Fund Name', 'Direct Consistency Ratio', 'Years of Existence', 'AUM (Rs. Crores)']:
                                if col in top_consistent.columns:
                                    available_columns.append(col)
                            
                            if available_columns:
                                st.dataframe(
                                    top_consistent[available_columns].round(3),
                                    use_container_width=True
                                )
                            else:
                                st.warning("No data available for display")
                        
                        with col2:
                            fig = px.bar(
                                top_consistent,
                                x='Direct Consistency Ratio',
                                y='Fund Name',
                                orientation='h',
                                title="Top 10 Most Consistent Funds (Direct)"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No funds found with valid Direct Consistency Ratio data")
                else:
                    st.warning("No consistency data available. This might be due to:")
                    st.write("- No funds meeting the minimum years requirement")
                    st.write("- Missing benchmark data for comparison")
                    st.write("- Insufficient return data for consistency calculation")
            
                # Consistency vs AUM analysis
                st.subheader("Consistency vs AUM Analysis")
                
                # Add explanation of how consistency is calculated
                with st.expander("ℹ️ How Consistency is Calculated"):
                    st.write(f"""
                    **Consistency Ratio Calculation Method:**
                    
                    The consistency ratio measures how often a fund outperforms its benchmark across different time periods.
                    
                    **Current Filter:** Only funds with at least {min_years_required} year{'s' if min_years_required > 1 else ''} of data are included in this analysis.
                    
                    **Step-by-step calculation:**
                    1. **Benchmark Returns**: For each time period (1Y, 3Y, 5Y, 10Y), we identify the benchmark return
                    2. **Outperformance Check**: For each fund, we compare its returns against the benchmark for each period
                    3. **Counting Outperformances**: We count how many times each fund beats its benchmark
                    4. **Consistency Ratio**: We divide the number of outperformance periods by the total number of periods with data
                    
                    **Formula:**
                    ```
                    Consistency Ratio = (Number of Periods Outperforming Benchmark) / (Total Periods with Data)
                    ```
                    
                    **Example:**
                    - Fund A outperforms benchmark in 3 out of 4 time periods
                    - Consistency Ratio = 3/4 = 0.75 (75% consistency)
                    
                    **Interpretation:**
                    - **0.0 to 0.25**: Low consistency (rarely outperforms)
                    - **0.25 to 0.50**: Below average consistency
                    - **0.50 to 0.75**: Above average consistency
                    - **0.75 to 1.0**: High consistency (frequently outperforms)
                    
                    **Why this matters:**
                    - Higher consistency suggests more reliable performance
                    - Lower consistency may indicate higher volatility or risk
                    - Helps identify funds that consistently deliver alpha
                    
                    **Filter Options:**
                    - **1 Year**: Includes all funds with at least 1 year of data (most inclusive)
                    - **3 Years**: Includes funds with at least 3 years of data (balanced view)
                    - **5 Years**: Includes funds with at least 5 years of data (more established funds)
                    - **10 Years**: Includes only funds with at least 10 years of data (longest track record)
                    """)
                
                valid_consistency = consistency_df.dropna(subset=['Direct Consistency Ratio', 'AUM (Rs. Crores)'])
                if len(valid_consistency) > 0:
                    # Normalize AUM for bubble sizes (scale between 10 and 50)
                    valid_consistency_copy = valid_consistency.copy()
                    min_aum = valid_consistency_copy['AUM (Rs. Crores)'].min()
                    max_aum = valid_consistency_copy['AUM (Rs. Crores)'].max()
                    if max_aum > min_aum:
                        valid_consistency_copy['Bubble Size'] = 10 + (valid_consistency_copy['AUM (Rs. Crores)'] - min_aum) / (max_aum - min_aum) * 40
                    else:
                        valid_consistency_copy['Bubble Size'] = 30  # Default size if all AUM values are same
                    
                    fig = px.scatter(
                        valid_consistency_copy,
                        x='Direct Consistency Ratio',
                        y='Years of Existence',
                        size='Bubble Size',
                        color='AUM (Rs. Crores)',
                        color_continuous_scale='Viridis',
                        hover_data=['Fund Name', 'AUM (Rs. Crores)'],
                        title="Consistency vs Years of Existence<br><sub>Bubble size represents AUM (Rs. Crores)</sub>",
                        labels={'Direct Consistency Ratio': 'Consistency Ratio', 'Years of Existence': 'Years of Existence'}
                    )
                    
                    # Apply dark theme styling
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis=dict(
                            gridcolor='rgba(128,128,128,0.3)',
                            linecolor='rgba(128,128,128,0.5)'
                        ),
                        yaxis=dict(
                            gridcolor='rgba(128,128,128,0.3)',
                            linecolor='rgba(128,128,128,0.5)'
                        ),
                        coloraxis_colorbar=dict(
                            title="AUM (Rs. Crores)",
                            title_font=dict(color='white'),
                            tickfont=dict(color='white')
                        ),
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlation analysis
                    correlation = valid_consistency['AUM (Rs. Crores)'].corr(valid_consistency['Direct Consistency Ratio'])
                    st.metric("Correlation between AUM and Consistency", f"{correlation:.3f}")
        
        elif analysis_type == "AUM Correlation":
            st.header("💰 AUM Correlation Analysis")
            
            # AUM vs Performance Correlation Matrix
            st.subheader("📊 AUM vs Performance Correlation Matrix")
            
            if 'Daily AUM (Cr.)' in df.columns:
                # Create correlation matrix data
                correlation_data = []
                aum_col = 'Daily AUM (Cr.)'
                
                for period in time_periods:
                    direct_col = f'Return {period} (%) Direct'
                    regular_col = f'Return {period} (%) Regular'
                    
                    if direct_col in df.columns:
                        # Direct plan correlation
                        direct_corr = df[aum_col].corr(df[direct_col])
                        correlation_data.append({
                            'Time Period': period,
                            'Plan Type': 'Direct',
                            'Correlation': direct_corr,
                            'Sample Size': len(df[[aum_col, direct_col]].dropna())
                        })
                    
                    if regular_col in df.columns:
                        # Regular plan correlation
                        regular_corr = df[aum_col].corr(df[regular_col])
                        correlation_data.append({
                            'Time Period': period,
                            'Plan Type': 'Regular',
                            'Correlation': regular_corr,
                            'Sample Size': len(df[[aum_col, regular_col]].dropna())
                        })
                
                if correlation_data:
                    corr_df = pd.DataFrame(correlation_data)
                    
                    # Display correlation table
                    st.write("**Correlation Coefficients (AUM vs Returns):**")
                    st.dataframe(corr_df.round(4), use_container_width=True)
                    
                    # Create correlation heatmap
                    pivot_corr = corr_df.pivot(index='Time Period', columns='Plan Type', values='Correlation')
                    
                    # Reorder the index to ensure correct time period sequence
                    pivot_corr = pivot_corr.reindex(time_periods)
                    
                    # Create custom colorscale for correlation values
                    def categorize_correlation_value(value):
                        if pd.isna(value):
                            return 0  # No data
                        elif value >= 0.6:
                            return 3  # Dark Green (≥0.6)
                        elif value >= 0.3:
                            return 2  # Orange (0.3 to 0.6)
                        else:
                            return 1  # Red (<0.3)
                    
                    # Convert correlation values to categories
                    categorical_corr = pivot_corr.applymap(categorize_correlation_value)
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=categorical_corr.values,
                        x=pivot_corr.columns,
                        y=pivot_corr.index,
                        colorscale=[
                            [0, '#DC143C'],      # Deep red for no data
                            [0.25, '#DC143C'],   # Deep red for no data
                            [0.26, '#8B0000'],   # Dark red for <0.3
                            [0.49, '#8B0000'],   # Dark red for <0.3
                            [0.5, '#FF8C00'],    # Orange for 0.3-0.6
                            [0.74, '#FF8C00'],   # Orange for 0.3-0.6
                            [0.75, '#2E8B57'],   # Dark green for ≥0.6
                            [1, '#2E8B57']       # Dark green for ≥0.6
                        ],
                        text=pivot_corr.round(3).values,
                        texttemplate="%{text}",
                        textfont={"size": 12, "color": "white"},
                        hoverongaps=False,
                        showscale=True,
                        colorbar=dict(
                            title="Correlation Level",
                            tickvals=[0, 1, 2, 3],
                            ticktext=["No Data", "<0.3", "0.3-0.6", "≥0.6"],
                            tickmode="array"
                        )
                    ))
                    
                    fig.update_layout(
                        title="AUM vs Performance Correlation Heatmap",
                        xaxis_title="Plan Type",
                        yaxis_title="Time Period",
                        width=400,  # Make chart smaller
                        height=300  # Make chart smaller
                    )
                    st.plotly_chart(fig, use_container_width=False)
                    
                    # Interpretation
                    st.subheader("📈 Correlation Interpretation")
                    st.write("""
                    **Correlation ranges:**
                    - **-1.0 to -0.7**: Strong negative correlation (larger funds = lower returns)
                    - **-0.7 to -0.3**: Moderate negative correlation
                    - **-0.3 to 0.3**: Weak or no correlation
                    - **0.3 to 0.7**: Moderate positive correlation
                    - **0.7 to 1.0**: Strong positive correlation (larger funds = higher returns)
                    """)
                    
                    # Find strongest correlations
                    strongest_pos = corr_df.loc[corr_df['Correlation'].idxmax()]
                    strongest_neg = corr_df.loc[corr_df['Correlation'].idxmin()]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Strongest Positive Correlation",
                            f"{strongest_pos['Time Period']} - {strongest_pos['Plan Type']}",
                            f"{strongest_pos['Correlation']:.4f}"
                        )
                    with col2:
                        st.metric(
                            "Strongest Negative Correlation",
                            f"{strongest_neg['Time Period']} - {strongest_neg['Plan Type']}",
                            f"{strongest_neg['Correlation']:.4f}"
                        )
                    
                    # Statistical significance (basic)
                    st.subheader("🔍 Statistical Insights")
                    significant_correlations = corr_df[abs(corr_df['Correlation']) > 0.3]
                    if len(significant_correlations) > 0:
                        st.write("**Moderate to Strong Correlations (|r| > 0.3):**")
                        st.dataframe(significant_correlations.round(4), use_container_width=True)
                    else:
                        st.write("**All correlations are weak (|r| < 0.3), suggesting no strong relationship between AUM size and performance.**")
                    
                    # Advanced correlation analysis with sample size considerations
                    st.subheader("📊 Advanced Correlation Analysis")
                    
                    # Calculate correlation strength categories
                    def categorize_correlation(corr, sample_size):
                        if pd.isna(corr) or pd.isna(sample_size):
                            return "Insufficient Data"
                        
                        abs_corr = abs(corr)
                        if sample_size < 10:
                            return "Very Small Sample"
                        elif abs_corr < 0.1:
                            return "Negligible"
                        elif abs_corr < 0.3:
                            return "Weak"
                        elif abs_corr < 0.5:
                            return "Moderate"
                        elif abs_corr < 0.7:
                            return "Strong"
                        else:
                            return "Very Strong"
                    
                    corr_df['Correlation Strength'] = corr_df.apply(
                        lambda x: categorize_correlation(x['Correlation'], x['Sample Size']), axis=1
                    )
                    
                    # Display enhanced correlation table
                    st.write("**Enhanced Correlation Analysis:**")
                    enhanced_corr = corr_df[['Time Period', 'Plan Type', 'Correlation', 'Sample Size', 'Correlation Strength']]
                    st.dataframe(enhanced_corr.round(4), use_container_width=True)
                    
                    # Summary statistics
                    st.subheader("📋 Correlation Summary")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        avg_corr = corr_df['Correlation'].mean()
                        st.metric("Average Correlation", f"{avg_corr:.4f}")
                    
                    with col2:
                        max_corr = corr_df['Correlation'].max()
                        st.metric("Highest Correlation", f"{max_corr:.4f}")
                    
                    with col3:
                        min_corr = corr_df['Correlation'].min()
                        st.metric("Lowest Correlation", f"{min_corr:.4f}")
                    
                    # Correlation patterns analysis
                    st.subheader("🔍 Pattern Analysis")
                    
                    # Check if correlations increase/decrease over time
                    time_trend = corr_df.groupby('Plan Type')['Correlation'].apply(list)
                    
                    for plan_type, correlations in time_trend.items():
                        if len(correlations) > 1:
                            # Simple trend: check if correlation increases or decreases over time
                            trend = "Increasing" if correlations[-1] > correlations[0] else "Decreasing"
                            st.write(f"**{plan_type} Plan Trend:** {trend} correlation over time periods")
                    
                    # AUM size categories correlation
                    st.subheader("🏗️ AUM Size Category Analysis")
                    
                    # Create AUM categories and analyze performance within each
                    if 'Daily AUM (Cr.)' in df.columns:
                        aum_data = df[['Daily AUM (Cr.)'] + [col for col in df.columns if 'Return' in col and 'Benchmark' not in col]].dropna()
                        
                        if len(aum_data) > 0:
                            # Categorize funds by AUM size
                            aum_quartiles = aum_data['Daily AUM (Cr.)'].quantile([0.25, 0.5, 0.75])
                            
                            def categorize_aum_size(aum):
                                if aum <= aum_quartiles[0.25]:
                                    return 'Small (≤25%)'
                                elif aum <= aum_quartiles[0.5]:
                                    return 'Medium-Small (25-50%)'
                                elif aum <= aum_quartiles[0.75]:
                                    return 'Medium-Large (50-75%)'
                                else:
                                    return 'Large (>75%)'
                            
                            aum_data['AUM Category'] = aum_data['Daily AUM (Cr.)'].apply(categorize_aum_size)
                            
                            # Analyze performance by AUM category for each time period
                            category_performance = []
                            for period in time_periods:
                                direct_col = f'Return {period} (%) Direct'
                                if direct_col in aum_data.columns:
                                    category_stats = aum_data.groupby('AUM Category')[direct_col].agg(['count', 'mean', 'std']).round(2)
                                    for category, stats in category_stats.iterrows():
                                        category_performance.append({
                                            'Time Period': period,
                                            'AUM Category': category,
                                            'Fund Count': stats['count'],
                                            'Avg Return (%)': stats['mean'],
                                            'Std Dev (%)': stats['std']
                                        })
                            
                            if category_performance:
                                category_df = pd.DataFrame(category_performance)
                                st.write("**Performance by AUM Size Category:**")
                                st.dataframe(category_df, use_container_width=True)
                                
                                # Visualize category performance
                                fig = px.box(
                                    aum_data,
                                    x='AUM Category',
                                    y=direct_col,
                                    title=f"Performance Distribution by AUM Size - {period}",
                                    color='AUM Category'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                
                else:
                    st.warning("No correlation data available. Please check if AUM and return columns contain valid numeric data.")
            
            # Original AUM analysis continues below
            st.subheader("📈 AUM Distribution Analysis")
            
            # AUM distribution
            st.subheader("AUM Distribution")
            
            if 'Daily AUM (Cr.)' in df.columns:
                aum_data = df['Daily AUM (Cr.)'].dropna()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(aum_data, title="AUM Distribution", nbins=20)
                    fig.update_layout(xaxis_title="AUM (Cr.)")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # AUM statistics
                    st.subheader("AUM Statistics")
                    st.metric("Average AUM", f"₹{aum_data.mean():.0f} Cr.")
                    st.metric("Median AUM", f"₹{aum_data.median():.0f} Cr.")
                    st.metric("Largest Fund", f"₹{aum_data.max():.0f} Cr.")
                    st.metric("Smallest Fund", f"₹{aum_data.min():.0f} Cr.")
                
                # Performance vs AUM correlation
                st.subheader("Performance vs AUM Correlation")
                
                selected_period_aum = st.selectbox("Select Period for AUM Analysis", time_periods)
                
                direct_col = f'Return {selected_period_aum} (%) Direct'
                
                if direct_col in df.columns:
                    corr_data = df[['Daily AUM (Cr.)', direct_col, 'Scheme Name']].dropna()
                    
                    if len(corr_data) > 0:
                        # Try to add trendline if statsmodels is available, otherwise create without it
                        try:
                            fig = px.scatter(
                                corr_data,
                                x='Daily AUM (Cr.)',
                                y=direct_col,
                                hover_data=['Scheme Name'],
                                title=f"AUM vs {selected_period_aum} Returns (Direct)",
                                trendline="ols"
                            )
                        except:
                            # Fallback without trendline if statsmodels is not available
                            fig = px.scatter(
                                corr_data,
                                x='Daily AUM (Cr.)',
                                y=direct_col,
                                hover_data=['Scheme Name'],
                                title=f"AUM vs {selected_period_aum} Returns (Direct) (No trendline - statsmodels not available)"
                            )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Correlation coefficient
                        correlation = corr_data['Daily AUM (Cr.)'].corr(corr_data[direct_col])
                        st.metric(f"Correlation (AUM vs {selected_period_aum} Returns)", f"{correlation:.3f}")
                        
                        # Size-based performance analysis
                        st.subheader("Size-Based Performance Analysis")
                        
                        # Categorize funds by AUM
                        aum_quartiles = corr_data['Daily AUM (Cr.)'].quantile([0.25, 0.5, 0.75])
                        
                        def categorize_aum(aum):
                            if aum <= aum_quartiles[0.25]:
                                return 'Small'
                            elif aum <= aum_quartiles[0.5]:
                                return 'Medium-Small'
                            elif aum <= aum_quartiles[0.75]:
                                return 'Medium-Large'
                            else:
                                return 'Large'
                        
                        corr_data['AUM Category'] = corr_data['Daily AUM (Cr.)'].apply(categorize_aum)
                        
                        # Box plot by AUM category
                        fig = px.box(
                            corr_data,
                            x='AUM Category',
                            y=direct_col,
                            title=f"Performance by AUM Category - {selected_period_aum}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Performance summary by AUM category
                        aum_performance = corr_data.groupby('AUM Category')[direct_col].agg(['count', 'mean', 'std']).round(2)
                        st.dataframe(aum_performance, use_container_width=True)
        
        elif analysis_type == "Performance vs Benchmark Heatmap":
            st.header("🔥 Performance vs Benchmark Heatmap")
            
            # Create performance vs benchmark heatmap
            st.subheader("Direct Plan Performance vs Benchmark")
            
            # Get benchmark returns for all periods
            benchmark_returns = {}
            for period in time_periods:
                benchmark_col = f'Return {period} (%) Benchmark'
                if benchmark_col in df.columns:
                    benchmark_data = df[benchmark_col].dropna()
                    if len(benchmark_data) > 0:
                        benchmark_returns[period] = benchmark_data.iloc[0]
                    else:
                        benchmark_returns[period] = None
                else:
                    benchmark_returns[period] = None
            
            # Create performance vs benchmark data
            heatmap_data = []
            for idx, row in df.iterrows():
                fund_name = row['Scheme Name']
                fund_performance = {}
                
                for period in time_periods:
                    direct_col = f'Return {period} (%) Direct'
                    if direct_col in df.columns and not pd.isna(row[direct_col]):
                        if benchmark_returns.get(period) is not None:
                            # Calculate outperformance vs benchmark
                            outperformance = row[direct_col] - benchmark_returns[period]
                            fund_performance[period] = outperformance
                        else:
                            fund_performance[period] = np.nan
                    else:
                        fund_performance[period] = np.nan
                
                # Only add funds that have data for at least one period
                if any(not pd.isna(val) for val in fund_performance.values()):
                    fund_performance['Scheme Name'] = fund_name
                    heatmap_data.append(fund_performance)
            
            if heatmap_data:
                # Create DataFrame for heatmap
                heatmap_df = pd.DataFrame(heatmap_data)
                heatmap_df = heatmap_df.set_index('Scheme Name')
                
                # Reorder columns to match time_periods order
                heatmap_df = heatmap_df[time_periods]
                
                # Create the heatmap with three colors: Dark Green (above 0), Dark Red (0 or below), Grey (no data)
                def categorize_performance(value):
                    if pd.isna(value):
                        return 0  # Missing data (Grey)
                    elif value > 0:
                        return 2  # Above 0% (Dark Green)
                    else:
                        return 1  # 0% or below (Dark Red)
                
                # Convert to categorical values
                categorical_data = heatmap_df.applymap(categorize_performance)
                
                # Replace NaN values with "N/A" for display
                display_values = heatmap_df.round(2).values
                display_values = np.where(pd.isna(display_values), "N/A", display_values)
                
                # Create the heatmap with three discrete colors
                fig = go.Figure(data=go.Heatmap(
                    z=categorical_data.values,
                    x=heatmap_df.columns,
                    y=heatmap_df.index,
                    colorscale=[
                        [0, '#808080'],      # Grey for missing data (0)
                        [0.33, '#808080'],   # Grey for missing data (0)
                        [0.34, '#8B0000'],   # Dark red for 0% or below (1)
                        [0.66, '#8B0000'],   # Dark red for 0% or below (1)
                        [0.67, '#2E8B57'],   # Dark green for above 0% (2)
                        [1, '#2E8B57']       # Dark green for above 0% (2)
                    ],
                    text=display_values,
                    texttemplate="%{text}",
                    textfont={"size": 16, "color": "white"},
                    hoverongaps=False,
                    showscale=True,
                    colorbar=dict(
                        title="Performance Level",
                        tickvals=[0, 1, 2],
                        ticktext=["No Data", "≤0%", ">0%"],
                        tickmode="array"
                    )
                ))
                
                fig.update_layout(
                    title="Direct Plan Performance vs Benchmark",
                    xaxis_title="Time Period",
                    yaxis_title="Scheme Name",
                    width=1200,  # Double the chart size
                    height=800   # Double the chart size
                )
                
                # Update x-axis to show correct time period order
                fig.update_xaxes(categoryorder='array', categoryarray=time_periods)
                
                st.plotly_chart(fig, use_container_width=False)
                
                # Show summary statistics
                st.subheader("📊 Performance Summary")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Count of funds outperforming benchmark
                    total_funds = len(heatmap_df)
                    outperforming_funds = (heatmap_df > 0).sum().sum()
                    st.metric("Total Outperformances", f"{outperforming_funds}")
                
                with col2:
                    # Average outperformance
                    avg_outperformance = heatmap_df.mean().mean()
                    st.metric("Average Outperformance", f"{avg_outperformance:.2f}%")
                
                with col3:
                    # Best single outperformance
                    best_outperformance = heatmap_df.max().max()
                    st.metric("Best Outperformance", f"{best_outperformance:.2f}%")
                
                # Show detailed data table
                st.subheader("📋 Detailed Performance Data")
                
                # Color coding explanation
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("🟢 **Dark Green = Above 0%** (Outperforming benchmark)")
                with col2:
                    st.markdown("🔴 **Dark Red = 0% or Below** (At or underperforming benchmark)")
                with col3:
                    st.markdown("⚫ **Grey = No Data** (Missing performance data)")
                
                st.write("**Positive values = Outperforming benchmark, Negative values = Underperforming benchmark**")
                
                # Round the data for display
                display_df = heatmap_df.round(2)
                st.dataframe(display_df, use_container_width=True)
                
                # Export functionality
                if st.button("📥 Download Performance Data"):
                    csv = display_df.to_csv()
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"performance_vs_benchmark_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No performance data available for heatmap visualization.")
        
        elif analysis_type == "Investment Calculator":
            st.header("💰 Investment Calculator - Rs. 1 Lakh Investment Analysis")
            
            # Calculate final values for Rs. 1 Lakh investment
            st.subheader("Final Value Analysis for Rs. 1 Lakh Investment")
            
            # Create investment calculator data - show funds in all periods where they have data
            investment_data = []
            
            for idx, row in df.iterrows():
                fund_name = row['Scheme Name']
                initial_investment = 100000  # Rs. 1 Lakh
                
                # Check each time period and create separate entries for each period with data
                for period in time_periods:
                    regular_col = f'Return {period} (%) Regular'
                    direct_col = f'Return {period} (%) Direct'
                    
                    # Check if we have data for this period
                    has_regular_data = regular_col in df.columns and not pd.isna(row[regular_col])
                    has_direct_data = direct_col in df.columns and not pd.isna(row[direct_col])
                    
                    # Only create entry if we have at least one type of data for this period
                    if has_regular_data or has_direct_data:
                        fund_data = {
                            'Fund Name': fund_name,
                            'AUM (Rs. Crores)': row.get('Daily AUM (Cr.)', np.nan),
                            'Investment Period': period,
                            'Years Invested': int(period.split()[0]),  # Extract number from "1 Year", "3 Year", etc.
                            'Initial Investment (Rs.)': initial_investment
                        }
                        
                        # Calculate final values for this specific period
                        final_value_regular = None
                        final_value_direct = None
                        
                        if has_regular_data:
                            years = int(period.split()[0])
                            annual_return_regular = row[regular_col] / 100
                            final_value_regular = initial_investment * ((1 + annual_return_regular) ** years)
                        
                        if has_direct_data:
                            years = int(period.split()[0])
                            annual_return_direct = row[direct_col] / 100
                            final_value_direct = initial_investment * ((1 + annual_return_direct) ** years)
                        
                        # Add calculated values to fund data
                        fund_data['Final Value Regular (Rs.)'] = final_value_regular
                        fund_data['Final Value Direct (Rs.)'] = final_value_direct
                        
                        # Calculate absolute gains
                        if final_value_regular is not None:
                            fund_data['Gain Regular (Rs.)'] = final_value_regular - initial_investment
                            fund_data['Gain Regular (%)'] = ((final_value_regular - initial_investment) / initial_investment) * 100
                        
                        if final_value_direct is not None:
                            fund_data['Gain Direct (Rs.)'] = final_value_direct - initial_investment
                            fund_data['Gain Direct (%)'] = ((final_value_direct - initial_investment) / initial_investment) * 100
                        
                        # Calculate direct plan advantage
                        if final_value_regular is not None and final_value_direct is not None:
                            fund_data['Direct Advantage (Rs.)'] = final_value_direct - final_value_regular
                            fund_data['Direct Advantage (%)'] = ((final_value_direct - final_value_regular) / final_value_regular) * 100
                        
                        # Add this fund-period combination to the data
                        investment_data.append(fund_data)
            
            if investment_data:
                investment_df = pd.DataFrame(investment_data)
                
                
                # Top performers by final value
                st.subheader("🏆 Top Performers by Final Value (Direct Plans)")
                st.write("**Note:** All calculations based on ₹1,00,000 initial investment")
                
                # Add filter for time period
                period_filter = st.selectbox(
                    "Filter by Investment Period",
                    ["All Periods", "1 Year", "3 Year", "5 Year", "10 Year"],
                    help="Filter funds by their investment period"
                )
                
                # Debug: Show available investment periods
                with st.expander("🔍 Debug: Available Investment Periods"):
                    if len(investment_df) > 0:
                        available_periods = investment_df['Investment Period'].value_counts()
                        st.write("**Available Investment Periods in Data:**")
                        st.dataframe(available_periods.to_frame('Fund Entries'), use_container_width=True)
                        
                        # Show unique funds vs total entries
                        unique_funds = investment_df['Fund Name'].nunique()
                        total_entries = len(investment_df)
                        st.write(f"**Data Summary:**")
                        st.write(f"- **Unique Funds:** {unique_funds}")
                        st.write(f"- **Total Fund-Period Entries:** {total_entries}")
                        st.write(f"- **Note:** Each fund can appear multiple times (once per time period with data)")
                    else:
                        st.write("No investment data available")
                
                # Filter data based on selected period
                if period_filter == "All Periods":
                    filtered_df = investment_df.dropna(subset=['Final Value Direct (Rs.)'])
                else:
                    filtered_df = investment_df[
                        (investment_df['Investment Period'] == period_filter) & 
                        (investment_df['Final Value Direct (Rs.)'].notna())
                    ]
                
                # Sort by direct plan final value - always show all funds
                top_direct = filtered_df.sort_values('Final Value Direct (Rs.)', ascending=False)
                display_title = f"All Funds by Final Value (Direct Plans) - {period_filter}"
                
                # Display dynamic summary metrics based on filtered data
                st.subheader("📊 Investment Summary")
                
                # Show the investment amount prominently
                st.info(f"💰 **Investment Amount:** ₹1,00,000 (Rs. 1 Lakh) per fund")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Count unique funds for the selected period
                    unique_funds_filtered = top_direct['Fund Name'].nunique() if len(top_direct) > 0 else 0
                    st.metric("Total Unique Funds", unique_funds_filtered)
                
                with col2:
                    if len(top_direct) > 0 and 'Final Value Direct (Rs.)' in top_direct.columns:
                        best_direct = top_direct['Final Value Direct (Rs.)'].max()
                        st.metric("Best Direct Plan Value", f"₹{best_direct:,.0f}")
                    else:
                        st.metric("Best Direct Plan Value", "N/A")
                
                with col3:
                    if len(top_direct) > 0 and 'Final Value Regular (Rs.)' in top_direct.columns:
                        best_regular = top_direct['Final Value Regular (Rs.)'].max()
                        st.metric("Best Regular Plan Value", f"₹{best_regular:,.0f}")
                    else:
                        st.metric("Best Regular Plan Value", "N/A")
                
                with col4:
                    if len(top_direct) > 0 and 'Direct Advantage (Rs.)' in top_direct.columns:
                        max_advantage = top_direct['Direct Advantage (Rs.)'].max()
                        st.metric("Max Direct Advantage", f"₹{max_advantage:,.0f}")
                    else:
                        st.metric("Max Direct Advantage", "N/A")
                
                if len(top_direct) > 0:
                    # Show summary of filtered results
                    st.info(f"📊 **Showing {len(top_direct)} fund entries** for {period_filter} period")
                    
                    # Display table with formatted values
                    display_columns = [
                        'Fund Name', 'Investment Period', 'Years Invested', 
                        'Initial Investment (Rs.)', 'Final Value Direct (Rs.)', 
                        'Gain Direct (Rs.)', 'Gain Direct (%)', 'AUM (Rs. Crores)'
                    ]
                    
                    # Format the display dataframe
                    display_df = top_direct[display_columns].copy()
                    
                    # Format currency columns
                    currency_columns = ['Initial Investment (Rs.)', 'Final Value Direct (Rs.)', 'Gain Direct (Rs.)']
                    for col in currency_columns:
                        if col in display_df.columns:
                            display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.0f}" if pd.notna(x) else "N/A")
                    
                    # Ensure Initial Investment shows as 1,00,000 for all rows
                    if 'Initial Investment (Rs.)' in display_df.columns:
                        display_df['Initial Investment (Rs.)'] = '₹1,00,000'
                    
                    # Format percentage columns
                    if 'Gain Direct (%)' in display_df.columns:
                        display_df['Gain Direct (%)'] = display_df['Gain Direct (%)'].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
                    
                    # Format AUM column
                    if 'AUM (Rs. Crores)' in display_df.columns:
                        display_df['AUM (Rs. Crores)'] = display_df['AUM (Rs. Crores)'].apply(lambda x: f"₹{x:,.0f} Cr" if pd.notna(x) else "N/A")
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Create visualization
                    fig = px.bar(
                        top_direct,
                        x='Final Value Direct (Rs.)',
                        y='Fund Name',
                        orientation='h',
                        title=display_title,
                        hover_data=['Investment Period', 'Years Invested', 'Gain Direct (%)']
                    )
                    
                    # Format x-axis as currency and adjust height based on number of funds
                    chart_height = max(400, len(top_direct) * 30)  # Dynamic height
                    fig.update_layout(
                        xaxis_tickformat='₹,.0f',
                        height=chart_height
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"⚠️ **No funds found** for {period_filter} period. Try selecting a different period or 'All Periods'.")
                
                # Direct vs Regular comparison
                st.subheader("📈 Direct vs Regular Plan Comparison")
                
                # Filter funds that have both regular and direct data
                comparison_funds = investment_df.dropna(subset=['Final Value Regular (Rs.)', 'Final Value Direct (Rs.)'])
                
                if len(comparison_funds) > 0:
                    # Create scatter plot
                    fig = px.scatter(
                        comparison_funds,
                        x='Final Value Regular (Rs.)',
                        y='Final Value Direct (Rs.)',
                        size='AUM (Rs. Crores)',
                        color='Direct Advantage (Rs.)',
                        hover_data=['Fund Name', 'Investment Period', 'Years Invested'],
                        title="Direct vs Regular Plan Final Values<br><sub>Bubble size represents AUM, Color represents Direct Advantage</sub>",
                        labels={
                            'Final Value Regular (Rs.)': 'Regular Plan Final Value (₹)',
                            'Final Value Direct (Rs.)': 'Direct Plan Final Value (₹)'
                        },
                        color_continuous_scale='RdYlGn'
                    )
                    
                    # Add diagonal line (where regular = direct)
                    min_val = min(comparison_funds['Final Value Regular (Rs.)'].min(), 
                                 comparison_funds['Final Value Direct (Rs.)'].min())
                    max_val = max(comparison_funds['Final Value Regular (Rs.)'].max(), 
                                 comparison_funds['Final Value Direct (Rs.)'].max())
                    fig.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, 
                                 line=dict(dash="dash", color="red", width=2))
                    
                    # Apply dark theme styling
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        xaxis=dict(
                            gridcolor='rgba(128,128,128,0.3)',
                            linecolor='rgba(128,128,128,0.5)',
                            tickformat='₹,.0f'
                        ),
                        yaxis=dict(
                            gridcolor='rgba(128,128,128,0.3)',
                            linecolor='rgba(128,128,128,0.5)',
                            tickformat='₹,.0f'
                        ),
                        coloraxis_colorbar=dict(
                            title="Direct Advantage (₹)",
                            title_font=dict(color='white'),
                            tickfont=dict(color='white'),
                            tickformat='₹,.0f'
                        ),
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_direct_advantage = comparison_funds['Direct Advantage (Rs.)'].mean()
                        st.metric("Average Direct Advantage", f"₹{avg_direct_advantage:,.0f}")
                    
                    with col2:
                        avg_direct_advantage_pct = comparison_funds['Direct Advantage (%)'].mean()
                        st.metric("Average Direct Advantage (%)", f"{avg_direct_advantage_pct:.2f}%")
                    
                    with col3:
                        funds_with_direct_advantage = (comparison_funds['Direct Advantage (Rs.)'] > 0).sum()
                        total_comparison_funds = len(comparison_funds)
                        st.metric("Funds with Direct Advantage", f"{funds_with_direct_advantage}/{total_comparison_funds}")
                
                # Investment period analysis
                st.subheader("📅 Investment Period Analysis")
                
                # Group by investment period
                period_analysis = investment_df.groupby('Investment Period').agg({
                    'Final Value Direct (Rs.)': ['count', 'mean', 'max'],
                    'Final Value Regular (Rs.)': ['count', 'mean', 'max'],
                    'Direct Advantage (Rs.)': 'mean'
                }).round(0)
                
                # Flatten column names
                period_analysis.columns = ['_'.join(col).strip() for col in period_analysis.columns]
                period_analysis = period_analysis.reset_index()
                
                # Display period analysis
                st.write("**Performance by Investment Period:**")
                st.dataframe(period_analysis, use_container_width=True)
                
                # Create period comparison chart
                if len(period_analysis) > 1:
                    fig = px.bar(
                        period_analysis,
                        x='Investment Period',
                        y=['Final Value Direct (Rs.)_mean', 'Final Value Regular (Rs.)_mean'],
                        title="Average Final Value by Investment Period",
                        barmode='group'
                    )
                    
                    fig.update_layout(
                        yaxis_tickformat='₹,.0f',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Export functionality
                st.subheader("📥 Export Investment Analysis")
                
                if st.button("Download Investment Calculator Data"):
                    # Prepare data for export
                    export_df = investment_df.copy()
                    
                    # Round numeric columns
                    numeric_columns = export_df.select_dtypes(include=[np.number]).columns
                    export_df[numeric_columns] = export_df[numeric_columns].round(2)
                    
                    csv = export_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"investment_calculator_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                # Methodology explanation
                with st.expander("ℹ️ Investment Calculator Methodology"):
                    st.write("""
                    **How the Investment Calculator Works:**
                    
                    **1. Investment Amount:** Fixed at ₹1,00,000 (Rs. 1 Lakh)
                    
                    **2. Data Selection:** For each fund, we use the longest available time period:
                    - If 10-year data is available, we use 10-year returns
                    - If not, we check 5-year, then 3-year, then 1-year data
                    - This ensures we show the maximum possible growth for each fund
                    
                    **3. Calculation Method:** Compound Interest Formula
                    ```
                    Final Value = Initial Investment × (1 + Annual Return)^Years
                    ```
                    
                    **4. Annual Return Calculation:**
                    - We convert the total return percentage to annual return
                    - For example: 50% over 3 years = (1.50)^(1/3) - 1 = 14.47% annually
                    
                    **5. Direct vs Regular Comparison:**
                    - Shows the advantage of choosing direct plans over regular plans
                    - Calculates both absolute (₹) and percentage advantages
                    
                    **6. Important Notes:**
                    - Returns are based on historical performance and don't guarantee future results
                    - Past performance doesn't indicate future performance
                    - This is for educational and comparison purposes only
                    - Actual returns may vary due to market conditions, fees, and other factors
                    
                    **7. Data Limitations:**
                    - Some funds may not have data for all time periods
                    - Newer funds will have shorter investment periods
                    - Returns are calculated based on available historical data
                    """)
            else:
                st.warning("No investment data available. Please check if return columns contain valid numeric data.")

        elif analysis_type == "Risk-Return Analysis":
            st.header("⚖️ Risk-Return Analysis")
            
            # Risk categorization
            st.subheader("Risk Category Distribution")
            
            if 'Riskometer Scheme' in df.columns:
                risk_dist = df['Riskometer Scheme'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(values=risk_dist.values, names=risk_dist.index, title="Risk Category Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.dataframe(risk_dist.to_frame('Count'), use_container_width=True)
                
                # Risk vs Return analysis
                st.subheader("Risk vs Return Analysis")
                
                selected_period_risk = st.selectbox("Select Period for Risk Analysis", time_periods, key="risk_period")
                
                direct_col = f'Return {selected_period_risk} (%) Direct'
                
                if direct_col in df.columns:
                    risk_return_data = df[['Riskometer Scheme', direct_col, 'Scheme Name', 'Daily AUM (Cr.)']].dropna()
                    
                    if len(risk_return_data) > 0:
                        # Box plot by risk category
                        fig = px.box(
                            risk_return_data,
                            x='Riskometer Scheme',
                            y=direct_col,
                            title=f"Returns by Risk Category - {selected_period_risk}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk category performance summary
                        risk_summary = risk_return_data.groupby('Riskometer Scheme')[direct_col].agg(
                            ['count', 'mean', 'std', 'min', 'max']
                        ).round(2)
                        st.dataframe(risk_summary, use_container_width=True)
        
        # Raw data view
        st.header("📋 Raw Data")
        with st.expander("View Raw Fund Data"):
            st.dataframe(df, use_container_width=True)
        
        # Export functionality
        st.header("📤 Export Analysis")
        
        if st.button("Generate Analysis Report"):
            # Create a summary report
            report_data = {
                'Total Funds': len(df),
                'Analysis Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Top Performer (1Y Direct)': df.loc[df['Return 1 Year (%) Direct'].idxmax(), 'Scheme Name'] if 'Return 1 Year (%) Direct' in df.columns and not df['Return 1 Year (%) Direct'].isna().all() else 'N/A'
            }
            
            st.success("Analysis complete! Key insights generated above.")
            
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.info("Please ensure the file has the correct format with headers in row 5")

else:
    st.info("👆 Please upload your Fund Performance Excel file to begin analysis")
    
    # Sample data format guide
    with st.expander("Expected File Format"):
        st.write("""
        The Excel file should contain:
        - Headers in row 5 (Scheme Name, Benchmark, etc.)
        - Fund data starting from row 6
        - Return columns for 1, 3, 5, and 10 years (Regular and Direct)
        - Daily AUM data
        - Risk category information
        """)
    
    # Optional dependencies note
    with st.expander("Optional Dependencies"):
        st.write("""
        **Enhanced Features (Optional):**
        - Install `statsmodels` for trendline analysis in scatter plots: `pip install statsmodels`
        - Without statsmodels, charts will display without trendlines but all other functionality remains intact
        """)