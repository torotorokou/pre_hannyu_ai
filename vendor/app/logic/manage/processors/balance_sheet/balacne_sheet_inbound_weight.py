def inbound_weight(df_receive):
    total = int(df_receive["正味重量"].sum())
    return total
