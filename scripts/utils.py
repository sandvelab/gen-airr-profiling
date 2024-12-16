def get_shared_region_type(df1, df2):
    junction_aa, cdr3_aa = False, False

    if 'junction_aa' in df1.columns and 'junction_aa' in df2.columns:
        if not any(df1.junction_aa.eq('') | df1.junction_aa.isna()) and not any(df2.junction_aa.eq('') | df2.junction_aa.isna()):
            junction_aa = True
    if 'cdr3_aa' in df1.columns and 'cdr3_aa' in df2.columns:
        if not any(df1.cdr3_aa.eq('') | df1.cdr3_aa.isna()) and not any(df2.cdr3_aa.eq('') | df2.cdr3_aa.isna()):
            cdr3_aa = True

    if junction_aa and cdr3_aa:
        return 'junction_aa'
    elif junction_aa and not cdr3_aa:
        return 'junction_aa'
    elif not junction_aa and cdr3_aa:
        return 'cdr3_aa'
    else:
        raise ValueError("No shared region type found")
