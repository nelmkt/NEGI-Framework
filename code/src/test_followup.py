"""Reviewer-requested contrasts and nuisance-model checks."""
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.sparse import csr_matrix
from fw_followup import dr_panel,caliper_effect,caliper_matrix,support_diagnostics
from fw_config import Config

def test_dr_recovers_effect_when_only_outcome_model_is_correct():
    rng=np.random.default_rng(809);n=25000;x=rng.normal(size=n)
    D=rng.binomial(1,expit(-1+.8*x+.6*(x*x-1)))
    X=np.c_[np.ones(n),x]
    y=2+3*x-2.5*D+rng.normal(size=n)
    r=dr_panel(y,D,X)
    assert abs(r["estimate"]+2.5)<.10
    assert abs(r["influence"].mean())<1e-5

def test_dr_recovers_effect_when_only_propensity_model_is_correct():
    rng=np.random.default_rng(811);n=35000;x=rng.normal(size=n)
    D=rng.binomial(1,expit(-.8+.7*x))
    X=np.c_[np.ones(n),x]
    y=2+2*x*x-2.5*D+rng.normal(size=n)
    r=dr_panel(y,D,X)
    assert abs(r["estimate"]+2.5)<.12

def test_caliper_effect_reweights_controls_and_drops_unobserved_pairs():
    y=np.array([1.,3.,5.,9.]);dose=np.array([1.,2.,0.,0.])
    A=csr_matrix([[0,0,1,1],[0,0,0,1]])
    ti=np.array([0,1])
    e,p=caliper_effect(y,dose,ti,A,np.ones(4))
    assert e==-6 and p==-3.6
    e,p=caliper_effect(y,dose,ti,A,np.array([1.,1.,1.,0.]))
    assert e==-4 and p==-4
    e,p=caliper_effect(y,dose,ti,A,np.array([1.,1.,0.,0.]))
    assert np.isnan(e) and np.isnan(p)

def test_calipers_require_each_covariate_and_minimum_controls():
    d=pd.DataFrame(dict(lst_pre=[0.,.1,.2,.8],ndvi_pre=[0.,.1,.2,0.],
        elev=[0.,.1,.2,0.],coast_km=[1.,1.1,1.2,1.]))
    t=np.array([True,False,False,False]);c=~t
    ti,A=caliper_matrix(d,t,c,pd.Series(["a"]*4),pd.Series(dict(lst_pre=1.,ndvi_pre=1.,elev=1.)),.5,Config(min_controls=2))
    assert ti.tolist()==[0]
    assert A.indices.tolist()==[1,2]
    ti,A=caliper_matrix(d,t,c,pd.Series(["a"]*4),pd.Series(dict(lst_pre=1.,ndvi_pre=1.,elev=1.)),.05,Config(min_controls=2))
    assert len(ti)==0

