# coding: utf-8
import pandas as pd
import asyncio
import utils

# ïîëó÷àåì ñïèñîê âñåõ õàáîâ
hubs = utils.parse_habr_hubs()
url_lst = hubs['URL'].tolist()  # ñïèñîê âñåõ url

# äîáàâëåíèå êîëè÷åñòâà ñòðàíèö
res = asyncio.run(utils.process_urls(url_lst))  # ïîëó÷àåì êîëè÷åñòâî ñòðàíèö
hubs.insert(5, 'Pages_cnt', res)  # äîáàâëÿåì êîëè÷åñòâî ñòðàíèö â äàòàôðåéì õàáîâ

# Cîõðàíÿåì äàòàôðåéì ñî ññûëêàìè íà õàáû
hubs.to_excel('hubs_urls.xlsx', index_label='ID')

# ## Ïàðñåð ññûëîê íà ñòàòüè âíóòðè õàáîâ

df_full = pd.DataFrame(columns=['Title', 'URL', 'Hub'])  # Ñîçäàíèå èòîãîâîãî DataFrame ñ õàáàìè

# Çàïóñê àñèíõðîííîé ôóíêöèè ñáîðà ñòàòåé âíóòðè õàáîâ
for i in range(len(hubs)):
    hub_url = hubs.iloc[i]['URL'] + 'articles/'
    df = asyncio.run(utils.parse_habr_articles_in_hub(hub_url, df_full))
    df_full = pd.concat([df_full, df], ignore_index=True)

# Ñìîòðèì ñêîëüêî âñåãî ñòàòåé ïîëó÷èëîñü ñîáðàòü
if df_full is not None:
    print(f"Âñåãî ñîáðàíî ñòàòåé: {len(df_full)}")


# Óáèðàåì äóáëèêàòû (åñëè îíè ãäå-òî ïðîáðàëèñü)
hubs_full = df_full.drop_duplicates(subset='URL')

# Ñîõðàíÿåì èòîãîâûé äàòàôðåéì ñî ññûëêàìè íà âñå ñòàòüè
hubs_full.to_parquet('hubs_to_articles_urls.parquet', index=False)

# Ðàçîáú¸ì íà 5 ÷àñòåé (äëÿ áîëåå ë¸ãêîé îáðàáîòêè îáùåãî ìàññèâà äàííûõ)
urls = hubs_full['URL']
hubs_parts = utils.split_list(urls, 5)

# Çàïóñêàåì îáùèé ñ÷¸ò÷èê è áëîêèðîâêó
global_counter = [0]
global_lock = asyncio.Lock()


# Çàïóñêàåì ôóíêöèþ îáðàáîòêè ñòàòåé
for i, urls_chunk in enumerate(hubs_parts, 1):
    print(f"Íà÷èíàåòñÿ îáðàáîòêà ÷àñòè {i} èç {len(hubs_parts)}")
    asyncio.run(utils.process_part(urls_chunk, i, global_counter, global_lock))


# Ñîçäà¸ì èòîãîâûé äàòàôðåéì èç 5 îòäåëüíûõ ôàéëîâ
fin_df = pd.DataFrame()
for part in range(1, len(hubs_parts)+1):
    df_part = pd.read_parquet(f'articles_part_{part}.parquet')
    fin_df = pd.concat([fin_df, df_part], ignore_index=True)


fin_df.info()

# Íàõîäèì ïîòåðÿííûå ïðè ïîëó÷åíèè ñòàòüè (ãäå áûë Semaphore = 50) URL
# Ñîõðàíÿåì èõ è äîáàâëÿåì â èòîãîâûé äàòàôðåéì
url_dif = list(set(urls) - set(fin_df['URL']))
missed_articles = asyncio.run(utils.parse_article(url_dif, global_counter, global_lock))
missed_articles.to_parquet('missed_articles.parquet', index=False)
fin_df = pd.concat([fin_df, missed_articles], ignore_index=True)

# Ñîõðàíåíèå èòîãîâ
fin_df.to_parquet('habr_articles_parsed_final.parquet', index=False)
