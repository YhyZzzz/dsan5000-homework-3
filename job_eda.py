"""Offline, reusable visual EDA for a cleaned job CSV; inputs are never modified."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
import numpy as np
import pandas as pd

from manual_cleaning import SKILLS

COLORS={'manual':'#52677f','llm':'#168786','accent':'#d6933c','unknown':'#c4cbd2'}
WORK=['remote','hybrid','onsite','conflicting','Unknown']
TYPE=['Full-time','Contract','Part-time','Internship','Temporary','Multiple','Other/unrecognized','Unknown']
FIELDS={'salary_annual_mid':'Annualized salary','experience_years_min':'Numeric experience',
        'work_arrangement':'Work arrangement','visa_sponsorship':'Visa sponsorship',
        'education_degrees':'Education','skills':'Shared skills / methods',
        'industry':'Industry','experience_level':'Title seniority'}
LIST_FIELDS=['skills','search_queries','job_types','education_degrees','benefits']
BOOL_FIELDS=['potential_non_job_posting','multiple_locations','salary_multiple_ranges',
             'salary_domain_outlier','experience_domain_outlier','salary_period_assumed',
             'salary_currency_assumed','quality_review']


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':13,
        'axes.labelsize':10,'figure.facecolor':'white','axes.facecolor':'white',
        'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white',
        'axes.titleweight':'bold','axes.labelcolor':'#354252','text.color':'#243343',
        'xtick.color':'#526170','ytick.color':'#526170'})


def flag(series):
    return series.map(lambda value:str(value).strip().lower()=='true').fillna(False)


def finish(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=165,bbox_inches='tight');return fig


def ordered_counts(series,order):
    return series.value_counts().reindex(order,fill_value=0)


def salary_filters(frame):
    """Nonexclusive exclusion flags; overlapping reasons must not be added."""
    numeric=frame[['salary_annual_min','salary_annual_max','salary_annual_mid']]
    return pd.DataFrame({
        'missing_or_nonfinite_pay':numeric.isna().any(axis=1)|~np.isfinite(numeric).all(axis=1),
        'not_USD_labelled':~frame['salary_currency'].eq('USD').fillna(False),
        'nonpositive_or_inverted_bounds':numeric.le(0).any(axis=1)|frame.salary_annual_min.gt(frame.salary_annual_max),
        'unresolved_salary_issue':frame['salary_issue'].notna(),
        'multiple_salary_bands':frame['salary_multiple_ranges'],
        'domain_outlier':frame['salary_domain_outlier'],
        'potential_survey_nonjob':frame['potential_non_job_posting'],
    },index=frame.index).fillna(False)


class JobEDA:
    """Load a cleaned CSV, validate keys/types, summarize it and plot its profile."""
    def __init__(self,csv_path,label,color,output_dir='reports/eda'):
        self.path=Path(csv_path);self.label=label;self.color=color;self.output=Path(output_dir)
        self.output.mkdir(parents=True,exist_ok=True)
        self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.frame=pd.read_csv(self.path)
        if self.frame.job_key.isna().any() or not self.frame.job_key.is_unique:
            raise ValueError('EDA requires nonmissing, unique job_key values.')
        for column in LIST_FIELDS:
            def decode(value):
                parsed=json.loads(value) if isinstance(value,str) else value
                if not isinstance(parsed,list) or any(not isinstance(x,str) for x in parsed):
                    raise ValueError('Expected a JSON string array in '+column)
                return sorted(set(parsed))
            self.frame[column]=self.frame[column].map(decode)
        for column in BOOL_FIELDS:self.frame[column]=flag(self.frame[column])
        for column in ['salary_annual_min','salary_annual_max','salary_annual_mid','experience_years_min','description_word_count']:
            self.frame[column]=pd.to_numeric(self.frame[column],errors='raise')
        for column in ['crawl_time_utc','posted_date']:
            self.frame[column]=pd.to_datetime(self.frame[column],utc=True,errors='coerce')
        self.frame['shared_skills']=self.frame.skills.map(lambda values:sorted(set(values)&set(SKILLS)))
        self.frame['shared_skill_count']=self.frame.shared_skills.map(len)
        self.frame['work_display']=self.frame.work_arrangement.fillna('Unknown')
        if not set(self.frame.work_display).issubset(WORK):raise ValueError('Unrecognized work-arrangement value.')
        self.frame['job_type_display']=self.frame.job_type.map(self.display_job_type)
        self.exclusions=salary_filters(self.frame)
        self.salary=self.frame.loc[~self.exclusions.any(axis=1)].copy()

    @staticmethod
    def display_job_type(value):
        if pd.isna(value):return 'Unknown'
        if value=='Contractor':return 'Contract'
        return value if value in TYPE else 'Other/unrecognized'

    def coverage(self,frame=None):
        frame=self.frame if frame is None else frame
        rows=[]
        for field,label in FIELDS.items():
            if field=='skills':present=frame.shared_skills.map(bool)
            elif field=='education_degrees':present=frame[field].map(bool)
            else:present=frame[field].notna()
            rows.append({'field':field,'label':label,'observed':int(present.sum()),
                         'denominator':len(frame),'coverage_percent':100*present.mean()})
        return pd.DataFrame(rows)

    def salary_summary(self):
        rows=[]
        samples={'all_annualized_USD':self.frame.loc[self.frame.salary_currency.eq('USD') & self.frame.salary_annual_mid.gt(0)],
                 'core_single_band':self.salary,
                 'core_explicit_period':self.salary.loc[~self.salary.salary_period_assumed],
                 'core_explicit_period_currency':self.salary.loc[~self.salary.salary_period_assumed & ~self.salary.salary_currency_assumed]}
        for name,frame in samples.items():
            values=frame.salary_annual_mid.dropna()
            rows.append({'filter':name,'n':len(values),'median':values.median(),'q25':values.quantile(.25),
                         'q75':values.quantile(.75),'mean':values.mean()})
        return pd.DataFrame(rows)

    def geography(self):
        eligible=self.frame.loc[self.frame.country.eq('United States') & self.frame.state.notna() & ~self.frame.multiple_locations]
        return eligible,eligible.state.value_counts().rename_axis('state').rename('postings').reset_index()

    def overview(self,slug):
        d=self.frame;fig,axes=plt.subplots(2,2,figsize=(11.8,7.9),layout='constrained')
        fig.suptitle(f'{self.label} | {len(d):,} postings',fontsize=17,fontweight='bold')
        ax=axes[0,0];values=self.salary.salary_annual_mid/1000
        ax.hist(values,bins=np.arange(0,601,25),color=self.color,edgecolor='white')
        if len(values):ax.axvline(values.median(),color=COLORS['accent'],lw=2,label=f'Median ${values.median():.1f}k')
        ax.set(title=f'Annualized salary | core n={len(values)}',xlabel='USD thousands / year',ylabel='Postings',xlim=(0,600))
        ax.legend(frameon=False,fontsize=9)
        for ax,column,order,title in [(axes[0,1],'work_display',WORK,'Work arrangement'),
                                      (axes[1,0],'job_type_display',TYPE,'Employment type')]:
            counts=ordered_counts(d[column],order);counts=counts[counts.gt(0)]
            colors=[COLORS['unknown'] if key in ['Unknown','Other/unrecognized'] else self.color for key in counts.index]
            bars=ax.barh(counts.index[::-1],counts.values[::-1]/len(d)*100,color=colors[::-1])
            ax.bar_label(bars,labels=[f'{v} ({v/len(d):.0%})' for v in counts.values[::-1]],padding=4,fontsize=9)
            ax.set(title=title,xlabel=f'% of all {len(d)} postings',xlim=(0,117))
        ax=axes[1,1];bins=np.geomspace(10,10000,25)
        ax.hist(d.description_word_count,bins=bins,color=self.color,edgecolor='white');ax.set_xscale('log')
        ax.set(title='Description length',xlabel='Words (log scale)',ylabel='Postings',xlim=(10,10000))
        return finish(fig,self.output/f'{slug}-overview.png')

    def run(self,slug):
        self.coverage().to_csv(self.output/f'{slug}-coverage.csv',index=False)
        self.salary_summary().to_csv(self.output/f'{slug}-salary-summary.csv',index=False)
        self.exclusions.sum().rename('excluded_postings').rename_axis('reason').to_csv(self.output/f'{slug}-salary-exclusions.csv')
        self.geography()[1].to_csv(self.output/f'{slug}-state-counts.csv',index=False)
        self.frame[['job_key','job_type','job_type_display','work_display','shared_skill_count']].to_csv(self.output/f'{slug}-display-normalization.csv',index=False)
        return self.overview(slug)


def matched_frames(manual,llm):
    a=manual.frame.set_index('job_key');b=llm.frame.set_index('job_key')
    if not set(b.index).issubset(a.index):raise ValueError('LLM file contains posting keys absent from the manual data.')
    a=a.loc[b.index].copy()
    for column in ['description_sha256','company_name','job_title']:
        if not a[column].equals(b[column]):raise ValueError('Source identity mismatch for '+column)
    return a,b


def salary_relationships(eda):
    d=eda.salary;fig,axes=plt.subplots(1,2,figsize=(11.8,4.7),layout='constrained')
    pairs=d.loc[d.experience_years_min.notna() & ~d.experience_domain_outlier]
    axes[0].scatter(pairs.experience_years_min,pairs.salary_annual_mid/1000,s=25,alpha=.48,color=eda.color,edgecolors='none')
    axes[0].set(title=f'Experience and advertised pay | n={len(pairs)}',xlabel='Minimum experience (years)',ylabel='Annualized salary midpoint (USD thousands)')
    valid=d.experience_level.value_counts();groups=[g for g in ['Entry/Junior','Senior','Principal/Staff/Lead','Manager','Director'] if valid.get(g,0)>=10]
    samples=[d.loc[d.experience_level.eq(g),'salary_annual_mid'].to_numpy()/1000 for g in groups]
    if samples:
        bp=axes[1].boxplot(samples,orientation='horizontal',tick_labels=[f'{g}\n(n={len(v)})' for g,v in zip(groups,samples)],patch_artist=True,medianprops={'color':'white','linewidth':2},flierprops={'markersize':3})
        for box in bp['boxes']:box.set_facecolor(eda.color)
    axes[1].set(title='Pay by title seniority | groups with n≥10',xlabel='Annualized salary midpoint (USD thousands)')
    summary=pairs[['salary_annual_mid','experience_years_min']].rank().corr().iloc[0,1]
    stats={'paired_postings':len(pairs),'spearman_salary_experience':float(summary)}
    (eda.output/'salary-experience.json').write_text(json.dumps(stats,indent=2)+'\n')
    return finish(fig,eda.output/'salary-relationships.png'),stats


def geography_plot(eda):
    eligible,counts=eda.geography();top=counts.head(10)
    fig,axes=plt.subplots(1,2,figsize=(11.8,5.2),layout='constrained')
    axes[0].barh(top.state[::-1],top.postings[::-1],color=eda.color)
    axes[0].set(title=f'State footprint | eligible n={len(eligible)}',xlabel='Postings (top 10 states / DC)')
    for i,(_,row) in enumerate(top.iloc[::-1].iterrows()):axes[0].text(row.postings+.5,i,str(row.postings),va='center',fontsize=9)
    axes[0].set_xlim(0,top.postings.max()*1.16)
    states=top.loc[top.postings.ge(15),'state'].head(8).tolist()
    table=pd.crosstab(eligible.state,eligible.work_display).reindex(index=states,columns=WORK,fill_value=0)
    table=table.div(table.sum(axis=1),axis=0)*100
    left=np.zeros(len(table));palette=['#168786','#81b6b1','#52677f','#d6933c','#d8dde3']
    for value,color in zip(WORK,palette):
        axes[1].barh(table.index,table[value],left=left,label=value,color=color);left+=table[value].to_numpy()
    axes[1].invert_yaxis();axes[1].set(title='Work labels within states | all labels retained',xlabel='% of eligible state postings',xlim=(0,100))
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=3,frameon=False,fontsize=9)
    table.to_csv(eda.output/'manual-state-work-percent.csv')
    return finish(fig,eda.output/'geography.png')


def skill_counts(frame):
    return Counter(value for values in frame.shared_skills for value in set(values))


def skills_plot(manual,llm,a,b):
    counts=skill_counts(manual.frame);top=[key for key,_ in counts.most_common(12)]
    fig,axes=plt.subplots(1,2,figsize=(11.8,5.4),layout='constrained')
    axes[0].barh(top[::-1],[counts[k]/len(manual.frame)*100 for k in top[::-1]],color=manual.color)
    axes[0].set(title=f'Mentioned skills / methods | manual n={len(manual.frame)}',xlabel='% of postings; mentions are nonexclusive')
    ca,cb=skill_counts(a),skill_counts(b)
    for y,key in enumerate(top[::-1]):
        x1,x2=ca[key]/len(a)*100,cb[key]/len(b)*100
        axes[1].plot([x1,x2],[y,y],color='#c2cbd3',lw=2)
        axes[1].scatter(x1,y,color=manual.color,s=42,label='Manual' if y==0 else None,zorder=3)
        axes[1].scatter(x2,y,color=llm.color,s=42,label='LLM' if y==0 else None,zorder=3)
    axes[1].set_yticks(range(len(top)),top[::-1]);axes[1].set(title=f'Same vocabulary and same {len(a)} postings',xlabel='% with a source-supported vocabulary mention')
    axes[1].legend(frameon=False,loc='lower right')
    for ax in axes:ax.set_xlim(0,105)
    table=pd.DataFrame([{'skill':key,'full_manual_count':counts[key],'matched_manual_count':ca[key],'matched_llm_count':cb[key],'matched_denominator':len(a)} for key in top])
    table.to_csv(manual.output/'skill-mentions.csv',index=False)
    return finish(fig,manual.output/'skills.png')


def query_skill_heatmap(eda):
    queries=['Data Scientist','Data Engineer','Machine Learning Engineer','Computer Vision Engineer',
             'NLP Engineer (Natural Language Processing)','Quantitative Analyst (Quant)',
             'Bioinformatics Data Scientist','Business Intelligence Analyst']
    names=['Data scientist','Data engineer','Machine learning','Computer vision','NLP','Quantitative analyst','Bioinformatics','Business intelligence']
    skills=[key for key,_ in skill_counts(eda.frame).most_common(9)]
    rows=[];labels=[]
    for query,name in zip(queries,names):
        subset=eda.frame.loc[eda.frame.search_queries.map(lambda values:query in values)]
        if not len(subset):continue
        labels.append(f'{name} (n={len(subset)})')
        counts=skill_counts(subset);rows.append([100*counts[k]/len(subset) for k in skills])
    values=np.asarray(rows);fig,ax=plt.subplots(figsize=(11.8,5.7),layout='constrained')
    im=ax.imshow(values,cmap='YlGnBu',vmin=0,vmax=100,aspect='auto')
    ax.set_xticks(range(len(skills)),skills,rotation=35,ha='right');ax.set_yticks(range(len(labels)),labels)
    ax.set_title('Skill / method mentions within search-query groups',pad=14)
    for y in range(len(labels)):
        for x in range(len(skills)):ax.text(x,y,f'{values[y,x]:.0f}%',ha='center',va='center',color='white' if values[y,x]>55 else '#243343',fontsize=9)
    fig.colorbar(im,ax=ax,label='% of postings within query group',shrink=.8)
    pd.DataFrame(values,index=labels,columns=skills).to_csv(eda.output/'query-skill-percent.csv')
    return finish(fig,eda.output/'query-skills.png')


def correlation_plot(eda):
    d=eda.frame[['salary_annual_mid','experience_years_min','shared_skill_count','description_word_count']].copy()
    d.loc[~d.index.isin(eda.salary.index),'salary_annual_mid']=np.nan
    d.loc[eda.frame.experience_domain_outlier,'experience_years_min']=np.nan
    columns=list(d);r=pd.DataFrame(np.nan,index=columns,columns=columns);counts=r.copy()
    for x in columns:
        for y in columns:
            paired=d[[x,y]].dropna() if x!=y else d[[x]].dropna()
            counts.loc[x,y]=len(paired)
            if len(paired)>=20:
                r.loc[x,y]=1.0 if x==y else paired.rank().corr().iloc[0,1]
    fig,ax=plt.subplots(figsize=(7.4,5.6),layout='constrained')
    im=ax.imshow(r.to_numpy(dtype=float),cmap='BrBG',vmin=-1,vmax=1)
    labels=['Salary\n(core)','Experience\n(years)','Shared skill\ncount','Description\nwords']
    ax.set_xticks(range(4),labels);ax.set_yticks(range(4),labels);ax.set_title('Pairwise Spearman associations | manual full dataset',pad=15)
    for y in range(4):
        for x in range(4):
            value=r.iloc[y,x];label=f'{value:.2f}\nn={int(counts.iloc[y,x])}' if pd.notna(value) else 'n<20'
            ax.text(x,y,label,ha='center',va='center',color='white' if pd.notna(value) and abs(value)>.65 else '#243343',fontsize=10)
    fig.colorbar(im,ax=ax,label='Spearman rank correlation',shrink=.8)
    r.to_csv(eda.output/'correlations-spearman.csv');counts.to_csv(eda.output/'correlations-pair-counts.csv')
    return finish(fig,eda.output/'correlations.png'),r,counts


def comparison_plot(manual,llm,a,b):
    coverage=pd.DataFrame({'label':list(FIELDS.values()),'manual_percent':manual.coverage(a).coverage_percent.to_numpy(),
                          'llm_percent':llm.coverage(b).coverage_percent.to_numpy()})
    fig,axes=plt.subplots(1,2,figsize=(11.8,5.1),layout='constrained')
    y=np.arange(len(coverage));axes[0].barh(y-.18,coverage.manual_percent,height=.34,color=manual.color,label='Manual')
    axes[0].barh(y+.18,coverage.llm_percent,height=.34,color=llm.color,label='LLM')
    axes[0].set_yticks(y,coverage.label);axes[0].invert_yaxis();axes[0].set(title=f'Field coverage | same {len(a)} postings',xlabel='Observed (%); coverage is not accuracy',xlim=(0,105))
    axes[0].legend(loc='lower right',frameon=False)
    common=manual.salary.set_index('job_key').index.intersection(llm.salary.set_index('job_key').index)
    x=a.loc[common,'salary_annual_mid'];z=b.loc[common,'salary_annual_mid']
    axes[1].scatter(x/1000,z/1000,s=32,color=llm.color,alpha=.7,edgecolors='white',lw=.4)
    axes[1].plot([0,600],[0,600],color='#8b98a4',ls='--',lw=1)
    axes[1].set(title=f'Core pay extracted by both | n={len(common)}',xlabel='Manual annual midpoint (USD thousands)',ylabel='LLM annual midpoint (USD thousands)',xlim=(0,600),ylim=(0,600))
    close=(x-z).abs().le((x.abs()*.01).clip(lower=1))
    axes[1].text(.04,.96,f'{int(close.sum())}/{len(common)} within 1% or $1',transform=axes[1].transAxes,va='top',fontsize=10)
    coverage.to_csv(manual.output/'matched-field-coverage.csv',index=False)
    pd.DataFrame({'manual':x,'llm':z,'difference':z-x,'within_tolerance':close}).to_csv(manual.output/'matched-salary-pairs.csv')
    transition=pd.crosstab(a.work_display,b.work_display).reindex(index=WORK,columns=WORK,fill_value=0)
    transition.to_csv(manual.output/'matched-work-label-transitions.csv')
    return finish(fig,manual.output/'matched-extractors.png'),coverage,transition
