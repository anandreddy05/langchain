from langchain_text_splitters import RecursiveCharacterTextSplitter
text = """
    Hello is a salutation or greeting in the English language. It is first attested in writing from 1826.[1]

Early uses
Hello, with that spelling, was used in publications in the U.S. as early as the 18 October 1826 edition of the Norwich Courier of Norwich, Connecticut.[1] Another early use was an 1833 American book called The Sketches and Eccentricities of Col. David Crockett, of West Tennessee,[2] which was reprinted that same year in The London Literary Gazette.[3] The word was extensively used in literature by the 1860s.[4]

Etymology
According to the Oxford English Dictionary, hello is an alteration of hallo, hollo,[1] which came from Old High German "halâ, holâ, emphatic imperative of halôn, holôn to fetch, used especially in hailing a ferryman".[5] It also connects the development of hello to the influence of an earlier form, holla, whose origin is in the French holà (roughly, 'whoa there!', from French là 'there').[6] As in addition to hello, halloo,[7] hallo, hollo, hullo and (rarely) hillo also exist as variants or related words, the word can be spelt using any of all five vowels.[8][9][10]

Telephone
Before the telephone, verbal greetings often involved a time of day, such as "good morning". When the telephone began connecting people in different time zones, greetings without time gained popularity.[11]

Thomas Edison is credited with popularizing hullo as a telephone greeting. In previous decades, hullo had been used as an exclamation of surprise (used early on by Charles Dickens in 1850)[12] and halloo was shouted at ferry boat operators by people who wanted to catch a ride.[13] According to one account, halloo was the first word Edison yelled into his strip phonograph when he discovered recorded sound in 1877.[12] Shortly after Alexander Graham Bell invented the telephone, he answered calls by saying "ahoy ahoy", borrowing the term used on ships.[13][14] There is no evidence the greeting caught on.[13] Edison suggested Hello! on August 15, 1877 in a letter to the president of Pittsburgh's Central District and Printing Telegraph Company, T. B. A. David:

Friend David, I do not think we shall need a call bell as Hello! can be heard 10 to 20 feet away. What you think? Edison – P.S. first cost of sender & receiver to manufacture is only $7.00.[12]

The first name tags to include Hello may have 1880 at Niagara Falls, which was the site of the first telephone operators convention. By 1889, central telephone exchange operators were known as "hello-girls" because of the association between the greeting and the telephone.[14][15]

A 1918 novel uses the spelling "Halloa" in the context of telephone conversations.[16]
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    length_function=len
)
res = splitter.split_text(text)
print(len(res))
print(res)