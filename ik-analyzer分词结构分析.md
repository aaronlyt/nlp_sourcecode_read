# 方案介绍

基于字符串匹配的分词方案（提供最长分词输出，和所有匹配输出两种），对于未登录词语和歧义处理效果不是很好

# 辅助数据存储结构

## package org.wltea.analyzer.dic.Hit

* 表示词典检索的命中结果
* 存储的字段有

    1. 匹配状态字段：完全匹配(MATCH), 不匹配(UNMATCH), 前缀匹配(PREFIX)
    
    2. 词典匹配过程中，当前匹配到的词典分支节点:     DictSegment           matchedDictSegment
    
    3. 词段的开始位置和结束位置：begin end

    4.  前缀匹配是指：待匹配字符串charArray中的字符和字典树中的一个字符    串的前n连续个字符完全匹配
    
## package org.wltea.analyzer.dic.DictSegment

* 字段（字典树存储结构、节点数据）
    
    1. 公用字典表，存储汉字 Map<Character , Character> charMap

    2. Map字典树存储结构　Map<Character , DictSegment>
    
    3. 数组字典树存储结构　DictSegment [],     **_++说明这是一个递归类，从而能够实现树结构的存储++_**

    4. ARRAY_LENGTH_LIMIT控制数组中存储内容的数目，节点的子节点数目小与    这个数目时，采用数组存储；否则将数组中的结构转移到Map结构中，并且    使用Map结构、将数组置空
    
    5. nodeState:当前DictSegment的状态，默认是0,１表示从根节点到当前节点的路径表示一个词语
    
    6. 当前节点上存储的字符　private Character nodeChar
    
    7. 当前节点存储的Segment数目　storeSize
 
   
*   辅助函数lookforSegment

    1.  private DictSegment lookforSegment(Character keyChar)
    2.  查找本节点下对应的KeyChar(charMap对应的value)的segment，如果没有    找到，则创   建新的segment
    3.  顺序
        1. 如果当前存储节点数目小与ARRAY_LENGTH_LIMIT，则首先查找数组，    如果没有找到，并且数组可以继续存储，则存储在数组中
        2. 否则将数组元素转移到Map结构中、并将数组置空
        3. 如果如果当前存储节点数目大于ARRAY_LENGTH_LIMIT，直接在Map结    构    中查找、存储
    
*   加载填充词典片段(function fillSegment)
    
    1. public synchronized void fillSegment(char[] charArray , int begin , int length)

    2. 是一个递归函数
    
    3. 执行的功能：
        >1.   如果charArray中的汉字没有在charMap中，则将其加入到字典中   
        >2.   搜索当前节点的存储，查询对应keyChar的keyChar，如果没有则创建(lookforSegment函数)   
        
        >3.   处理字对应的segment，如果次元没有完全加入词典树，则递归，如果是词元的最后一个char，设置当前节点状态为１，表明是一个完整的词(也就是这个词语已经完全存储在字典树中了)

* 匹配词段(function match)

    1. public Hit match(char[] charArray , int begin , int length , Hit searchHit)
    
    2. 执行的功能步骤
        
        >   在字典树中查找字对应的DictSegment，如果数组存储结构非空，先查找数组，否则查找Map存储结构
        
        >   如果查找成功，并且charArray中仍然存在没有查找的字符，递归
        
        >   正确匹配到charArray中的最后一个字符，字典树中的字符的nodeSate值为１，则是完全匹配；字典树中的匹配字符节点仍有后续节点，则是前前缀匹配(设置Hit结构)
        
        >   否则就是不匹配
        
        >   匹配过程中会记录匹配的开始位置、结束位置（通过Hit结构）


## package org.wltea.analyzer.dic.Dictionary

* 词典管理类,单例模式

* 设置分词器默认字典路径：　main.dic surname.dic quantifier.dic suffix.dic preposition.dic stopword.dic

*   词典对象(class DictSegment)
    1. 主词典对象, private DictSegment _MainDict
    2. 姓氏词典对象 private DictSegment _SurnameDict
    3. 量词词典对象　private DictSegment _QuantifierDict
    4. 后缀词典对象　private DictSegment _SuffixDict
    5. 副词，介词词典对象　private DictSegment _PrepDict
    6. 停用词集合词典对象　private DictSegment _StopWords

*   初始化系统词典
    
    >   类对象构造的时候便加载、初始化系统词典，构造所有词典树

    >   加载主词典以及扩展词典(function loadMainDict)
    
    >>1.  private void loadMainDict();
    >>2.  主要执行功能：
    >>>1.   读取main.dic文件内容，调用函数fillSegment构建主词典字典树
    >>>2.   通过Configuration类对象，读取扩展词典路径，重复上述步骤
    
    >   其他词典的加载方式同上
    
##   package org.wltea.analyzer.cfg.Configuration 

*   配置的文件路径

    1.  分词器配置文件路径
    2.  配置属性，扩展词典(这个文件中包含扩展词典文件的路径)
    3.  配置属性，扩展停用词词典

*   初始化配置文件: 获取扩展词典文件路径/扩展停用词文件路径的list

*   初始化子分词器实现
  
    1.  public static List<ISegmenter> loadSegmenter()
    2.  初始化词典实例，Dictionary.getInstance()，++**会构造所有的字典     树**++
    3.  返回包含处理数量词的子分词器（QuantifierSegmenter类）、处理中文     词的子分词器(CJKSegmenter类)、处理字母的子分词器(LetterSegmente     r)的列表
    
#   分词器

##  package org.wltea.analyzer.seg.ISegmenter

*   public interface ISegmenter
    >   子分词器接口
*   void nextLexeme(char[] segmentBuff , Context context)
    >   从分析器读取下一个可能分解的词元对象
*   void reset()
    >   重置子分析器状态

##  package org.wltea.analyzer.Lexeme;

*   语义单元(词元)；在Context类中使用

*   public final class Lexeme implements Comparable<Lexeme>
    >   支持排序

*   词元类型
    >1. 普通词元　TYPE_CJK_NORMAL = 0
    >2. 姓氏    TYPE_CJK_SN = 1
    >3. 尾缀    TYPE_CJK_SF = 2
    >4. 未知的  TYPE_CJK_UNKNOWN = 3
    >5. 数词    TYPE_NUM = 10
    >6. 量词    TYPE_NUMCOUNT = 11
    >7. 英文    TYPE_LETTER = 20

*   词元的属性
    >1. 词元的起始位移 private int offset
    >2. 词元的相对起始位置 private int begin
    >3. 词元的长度　private int length
    >4. 词元的文本  private String lexemeText
    >5. 词元类型    private int lexemeType

*   函数　public boolean equals(Object o);
    > 判断词元是否相等，起始位置偏移、起始位置、终止位置完全相同

*   函数  public int compareTo(Lexeme other);
    >1. 词元在排序集合中的比较算法
    >2. 比较优先级：　起始位置、词元长度

*   函数　public boolean isOverlap(Lexeme other)
    >1. 判断词元是否彼此包含（完全包含，使用起始位置、终止位置）

*   词元在文本中的位置函数
    >1. 在文本中的起始位置: offset + begin
    >2. 在文本中的结束位置：offset + begin + length

##  package org.wltea.analyzer.Context

### 内部类private class IKSortedLinkSet

*   关于Lexeme的双向链表，按照顺序排列
*   功能函数
    >1. private void addLexeme(): 加入到链表中,***++较长的词在前面++***
    >2. private Lexeme pollFirst()：取出链表集合的第一个元素
    >3. private Lexeme pollLast(): 取出链表集合的最后一个元素
    >4. private voidexcludeOverlap()：剔除相邻元素完全包含关系的lexemw,     进行最大切分的时候，过滤长度较小的交叠词元

### Context类解析

*   属性(待处理字符串、处理中的缓冲区的相关状态)
    
    ```
    //是否使用最大词长切分（粗粒度）
    private boolean isMaxWordLength = false;	
    //记录Reader内已分析的字串总长度
    //在分多段分析词元时，该变量累计当前的segmentBuff相对于reader的位移
    //已经处理的字符串的总长度
    private int buffOffset;	
    //最近一次读入的,可处理的字串长度
    private int available;
    //最近一次分词器分析的字串长度
    private int lastAnalyzed;	
    //当前缓冲区位置指针
    private int cursor; 
    //字符窜读取缓冲
    private char[] segmentBuff;
    /*
     * 记录正在使用buffer的分词器对象
     * 如果set中存在有分词器对象，则buffer不能进行位移操作（处于locked状态）
     */
    private Set<ISegmenter> buffLocker;
    /*
     * 词元结果集，为每次游标的移动，存储切分出来的词元
     */
    private IKSortedLinkSet lexemeSet;
    ```

##  package org.wltea.analyzer.IKSegmentation

* 对外提供服务的接口，实现分词处理的主流程，调用具体的分词器

* 成员变量
    
    ```
    private Reader input;	
	//默认缓冲区大小
	private static final int BUFF_SIZE = 3072;
	/*
	  *缓冲区耗尽的临界值
	  *每次保留的字符串长度作为下一次分词处理字符串的上下文信息
	  *比如每次读入3072个字符，但是保留48个字符不处理，这48个字符串
	  *做为下一次分词处理字符串的开始
	/
	private static final int BUFF_EXHAUST_CRITICAL = 48;	
    //字符窜读取缓冲
    private char[] segmentBuff;
	//分词器上下文，类成员变量，分词过程中存储相关信息的重要数据结构
	private Context context;
	//分词处理器列表
	private List<ISegmenter> segmenters;
    
    ```
* 主流程函数　next
    
    >1. 解析
        >>1.    ++==函数每次只处理一个buffer的字符串==++，不是处理完整的输入数据         ，所以需要BUFF_EXHAUST_CRITICAL做为下一次读入字符串的上下文信息
        >>2.    Context的类变量context保留处理字符串过程中的相关信息
            >>>1.   context.setCursor(buffIndex) 设置缓冲区内被处理字符的位置
            >>>2.   ontext.setLastAnalyzed(analyzedLength)记录被处理的字符串         的长度(一次buffer)
            >>>3.   context.setBuffOffset记录所有处理的的字符串的长度(所有处         理的字符串的长度，完整的输入层次)
    >2. 代码流程
        >>1.    读入一个buffer的数据进行处理
        >>2.    对于buffer中的每一个字符，首先进行规格化，然后遍历子分词器处理该         字符，判断该字符是前面前缀词的一部分、单独成词、未识别的字、后面         词语的开始
        >>3.    判断是否结束(保留固定数目的上下文)，跳出循环
        >>4.    重置分词器数据结构
        >>5.   设置context数据结构中的数据
        
    >3. 源码
    
    ```
    public synchronized Lexeme next() throws IOException {
		if(context.getResultSize() == 0){
			/*
			 * 从reader中读取数据，填充buffer
			 * 如果reader是分次读入buffer的，那么buffer要进行移位处理
			 * 移位处理上次读入的但未处理的数据
			 */
			//　只读入一次buffer，说明并不是一次处理完成所有的字符
            //　这个函数中context是一个类成员变量，保留多次执行该函数的中间信息
			int available = fillBuffer(input);
			
            if(available <= 0){
            	context.resetContext();
                return null;
            }else{
            	//分词处理
            	int analyzedLength = 0;
        		for(int buffIndex = 0 ; buffIndex < available ;  buffIndex++){
        			//移动缓冲区指针
        			context.setCursor(buffIndex);
        			//进行字符规格化（全角转半角，大写转小写处理,　每次只规格化当前的字符
        			segmentBuff[buffIndex] = CharacterHelper.regularize(segmentBuff[buffIndex]);
        			//遍历子分词器
     
        			for(ISegmenter segmenter : segmenters){
        				segmenter.nextLexeme(segmentBuff , context);
        			}
        			analyzedLength++;
        			/*
        			 * 满足一下条件时，
        			 * 1.available == BUFF_SIZE 表示buffer满载
        			 * 2.buffIndex < available - 1 && buffIndex > available - BUFF_EXHAUST_CRITICAL表示当前指针处于临界区内
        			 * 3.!context.isBufferLocked()表示没有segmenter在占用buffer
        			 * 要中断当前循环（buffer要进行移位，并再读取数据的操作）
        			 */
        			/*
        			 * 表示buffer满载的情况下，剩余BUFF_EXHAUST_CRITICAL个字符和下次读入的数据一起处理，
        			 * 也就是保留BUFF_EXHAUST_CRITICAL个字符作为context字符信息
        			 */
					//　buffer不满载的情况下直接处理
        			if(available == BUFF_SIZE
        					&& buffIndex < available - 1   
        					&& buffIndex > available - BUFF_EXHAUST_CRITICAL
        					&& !context.isBufferLocked()){

        				break;
        			}
        		}//for loop
				
				for(ISegmenter segmenter : segmenters){
					segmenter.reset();
				}
        		//System.out.println(available + " : " +  buffIndex);
            	//记录最近一次分析的字符长度
        		context.setLastAnalyzed(analyzedLength);
            	//同时累计已分析的字符长度(所有的已经处理的文本中的字符的长度)
        		context.setBuffOffset(context.getBuffOffset() + analyzedLength);
        		//如果使用最大切分，则过滤交叠的短词元
        		if(context.isMaxWordLength()){
        			context.excludeOverlap();
        		}
            	//读取词元池中的词元
            	return buildLexeme(context.firstLexeme());
            }//else
		}else{
			//读取词元池中的已有词元
			return buildLexeme(context.firstLexeme());
		}	
	}
    ```
*   数据读入函数　fillBuffer

    >1. private int fillBuffer(Reader reader) throws IOException
    >2. 首次读入(开始一次新的InputStream)，先读取BUFF_SIZE个字符串
    >3. 否则的话，现将上次读入进buffer但是没有处理的offset个(代码实现                中是使用context.getAvailable() - context.getLastAnalyzed()计算               )字符串移动到缓冲区前面，然后再读入BUFF_SIZE - offset个字符串
    >4. 设置context结构的available变量的值
    
    >5. 代码
    ```
        /**
         * 根据context的上下文情况，填充segmentBuff 
         * @param reader
         * @return 返回待分析的（有效的）字串长度
         * @throws IOException 
         */
        private int fillBuffer(Reader reader) throws IOException{
        	int readCount = 0;
        	if(context.getBuffOffset() == 0){
        		//首次读取reader
        		readCount = reader.read(segmentBuff);
        	}else{
        		int offset = context.getAvailable() - context.getLastAnalyzed();
        		if(offset > 0){
        			//最近一次读取的>最近一次处理的，将上一次读入但是未处理的字串拷贝到segmentBuff头部
        			System.arraycopy(segmentBuff , context.getLastAnalyzed() , this.segmentBuff , 0 , offset);
        			readCount = offset;
        		}
        		//继续读取reader ，以onceReadIn - onceAnalyzed为起始位置，继续填充segmentBuff剩余的部分
        		readCount += reader.read(segmentBuff , offset , BUFF_SIZE - offset);
        	}            	
        	//记录最后一次从Reader中读入的可用字符长度
        	context.setAvailable(readCount);
        	return readCount;
        }	
    ```

## package org.org.wltea.analyzer.seg.CJKSegmenter

*   中文（CJK）词元处理子分词器，涵盖一下范围
   
    >1. 中文词语
    >2. 姓名
    >3. 地名
    >4. 未知词（单字切分）
    >5. 日文/韩文（单字切分）

*   ==*判断方法*：对应字典树匹配，最终是调用DictSegment类中的match方法==
   
*   未识别词段处理函数(会处理姓氏词语和后缀词语) processUnKnown

    >1. private void processUnknown(char[] segmentBuff , Context context ,       int uBegin , int uEnd)
    >2. 执行功能、流程
        >>1.    单字符判断是否是介或副词，如果不是，判断是否是姓氏，如果是加         入到Lexeme List中(词的List)
        >>2.    将每一个单字加入到Lexeme List中
        >>3.    判断uEnd位置的词语是否是介词或者副词，如果不是，判断buffer字         符串uEnd后面的词语是否是后缀，如果是，加入到匹配词语
    >3. 源代码
    
    ```
    private void processUnknown(char[] segmentBuff , Context context , int uBegin , int uEnd){
		Lexeme newLexeme = null;
		
		Hit hit = Dictionary.matchInPrepDict(segmentBuff, uBegin, 1);		
		if(hit.isUnmatch()){//不是副词或介词			
			if(uBegin > 0){//处理姓氏
				hit = Dictionary.matchInSurnameDict(segmentBuff, uBegin - 1 , 1);
				if(hit.isMatch()){
					//输出姓氏
					newLexeme = new Lexeme(context.getBuffOffset() , uBegin - 1 , 1 , Lexeme.TYPE_CJK_SN);
					context.addLexeme(newLexeme);		
				}
			}			
		}
		
		//以单字输出未知词段
		for(int i = uBegin ; i <= uEnd ; i++){
			newLexeme = new Lexeme(context.getBuffOffset() , i , 1  , Lexeme.TYPE_CJK_UNKNOWN);
			context.addLexeme(newLexeme);		
		}
		
		hit = Dictionary.matchInPrepDict(segmentBuff, uEnd, 1);
		if(hit.isUnmatch()){//不是副词或介词
			int length = 1;
			while(uEnd < context.getAvailable() - length){//处理后缀词
				hit = Dictionary.matchInSuffixDict(segmentBuff, uEnd + 1 , length);
				if(hit.isMatch()){
					//输出后缀
					newLexeme = new Lexeme(context.getBuffOffset() , uEnd + 1  , length , Lexeme.TYPE_CJK_SF);
					context.addLexeme(newLexeme);
					break;
				}
				if(hit.isUnmatch()){
					break;
				}
				length++;
			}
		}		
	}
    ```
*   ++***词元处理分词器函数***++
    
    >1. public void nextLexeme(char[] segmentBuff , Context context)

    >2. 未识别，是指没有在字典中出现
    
    >3. 每次处理一个字符，**++==对于一个新的字符，可能出现的情况==++**
        >>1.    和前面的前缀词进行组合
            >>1.    和前面的前缀匹配词语组成一个完整匹配词语
            >>2.    和前面的前缀匹配词语组成一个前缀匹配词
            >>3.    上面都不成功
        >>2.    作为一个新的词语的开始
            >>>1.   一个单独的词语
            >>>2.   一个词语的前缀词语
            >>>3.   未识别的字

    >4. private List<Hit> hitList
        >>1.    Hit对列，记录匹配中的Hit对象,将前缀匹配加入进去(可能是一个完全匹         配，同时满足前缀匹配的情况)
        >>2.    词语匹配会在hitList中匹配前缀的基础，把后面连续的字加进去，继续匹         配字典树
        >>3.    当后面一个连续的字没有办法组成一个词语并且没有办法组成一个前缀匹         配词语时，将该前缀匹配从hitList中移除
        
    >5. private int doneIndex
        >>1.    已完成处理的位置
        >>2.    donIndex被重新赋值的情况
            >>>1.   和前缀词匹配成功时
            >>>2.   作为新词的开始，被判断是一个完整的词语
            >>>3.   作为一个新词的开始，被判断为未识别的词语
            >>>4.   遇到非中文字符
    >6. 未识别词段处理函数(processUnknown)调用
        >>1.    在每次需要做处理时，比如完全匹配，判断这两次处理之间是否存在未识         别的词语，如果存在则调用处理
    >7. 处理遇到非中文字符，就会清空hitList
    
    >8. 有些词语既是完全匹配词又是前缀词，比如字典中存在结合　结合上皮，那么便满     足这个情况

*   词元处理分词器函数源代码
    
    ```
    public void nextLexeme(char[] segmentBuff , Context context) {

		//读取当前位置的char	
		char input = segmentBuff[context.getCursor()];
		
		if(CharacterHelper.isCJKCharacter(input)){//是（CJK）字符，则进行处理
			if(hitList.size() > 0){
				//处理词段队列
				Hit[] tmpArray = hitList.toArray(new Hit[hitList.size()]);
				for(Hit hit : tmpArray){
				    // 从前往后匹配，会在之前前缀匹配的基础上继续进行匹配，尝试匹配出在字典中的词语--liyantao
					hit = Dictionary.matchWithHit(segmentBuff, context.getCursor() , hit);
					
					if(hit.isMatch()){//匹配成词
						//判断是否有不可识别的词段
						// liyantao 中间有不在词典中的词语
						if(hit.getBegin() > doneIndex + 1){
							//输出并处理从doneIndex+1 到 seg.start - 1之间的未知词段
							processUnknown(segmentBuff , context , doneIndex + 1 , hit.getBegin()- 1);
						}
						//输出当前的词
						Lexeme newLexeme = new Lexeme(context.getBuffOffset() , hit.getBegin() , context.getCursor() - hit.getBegin() + 1 , Lexeme.TYPE_CJK_NORMAL);
						context.addLexeme(newLexeme);
						//更新goneIndex，标识已处理
						if(doneIndex < context.getCursor()){
							doneIndex = context.getCursor();
						}
						//　既是前缀词又是完全词语，是存在的，比如词典中存在结合　结合上皮--liyantao
						if(hit.isPrefix()){//同时也是前缀
							
						}else{ //后面不再可能有匹配了
							//移出当前的hit
							hitList.remove(hit);
						}
						
					}else if(hit.isPrefix()){//前缀，未匹配成词
						
					}else if(hit.isUnmatch()){
					    //  liyantao 不匹配，后面不再可能有匹配了
						//移出当前的hit
						hitList.remove(hit);
					}
				}
			}
			
			//处理以input为开始的一个新hit，最为一个新的词语的开始
			Hit hit = Dictionary.matchInMainDict(segmentBuff, context.getCursor() , 1);
			if(hit.isMatch()){//匹配成词
				//判断是否有不可识别的词段
				if(context.getCursor() > doneIndex + 1){
					//输出并处理从doneIndex+1 到 context.getCursor()- 1之间的未知
					processUnknown(segmentBuff , context , doneIndex + 1 , context.getCursor()- 1);
				}
				//输出当前的词
				Lexeme newLexeme = new Lexeme(context.getBuffOffset() , context.getCursor() , 1 , Lexeme.TYPE_CJK_NORMAL);
				context.addLexeme(newLexeme);
				//更新doneIndex，标识已处理
				if(doneIndex < context.getCursor()){
					doneIndex = context.getCursor();
				}

				//  既是前缀词又是完全词语，是存在的，比如词典中存在结合　结合上皮－－liyantao
				if(hit.isPrefix()){//同时也是前缀
					//向词段队列增加新的Hit
					hitList.add(hit);
				}
				
			}else if(hit.isPrefix()){//前缀，未匹配成词
				//向词段队列增加新的Hit
				hitList.add(hit);
				
			}else if(hit.isUnmatch()){//不匹配，当前的input不是词，也不是词前缀，将其视为分割性的字符
				if(doneIndex >= context.getCursor()){
					//当前不匹配的字符已经被处理过了，不需要再processUnknown
					return;
				}
				
				//输出从doneIndex到当前字符（含当前字符）之间的未知词
				processUnknown(segmentBuff , context , doneIndex + 1 , context.getCursor());
				//更新doneIndex，标识已处理
				doneIndex = context.getCursor();
			}
			
		}else {//输入的不是中文(CJK)字符
			if(hitList.size() > 0
					&&  doneIndex < context.getCursor() - 1){
				for(Hit hit : hitList){
					//判断是否有不可识别的词段
					if(doneIndex < hit.getEnd()){
						//输出并处理从doneIndex+1 到 seg.end之间的未知词段
						processUnknown(segmentBuff , context , doneIndex + 1 , hit.getEnd());
					}
				}
			}
			//清空词段队列
			hitList.clear();
			//更新doneIndex，标识已处理
			if(doneIndex < context.getCursor()){
				doneIndex = context.getCursor();
			}
		}
		
		//缓冲区结束临界处理
		if(context.getCursor() == context.getAvailable() - 1){ //读取缓冲区结束的最后一个字符			
			if( hitList.size() > 0 //队列中还有未处理词段
				&& doneIndex < context.getCursor()){//最后一个字符还未被输出过
				for(Hit hit : hitList){
					//判断是否有不可识别的词段
					if(doneIndex < hit.getEnd() ){
						//输出并处理从doneIndex+1 到 seg.end之间的未知词段
						processUnknown(segmentBuff , context , doneIndex + 1 , hit.getEnd());
					}
				}
			}
			//清空词段队列
			hitList.clear();;
		}
		
		//判断是否锁定缓冲区
		if(hitList.size() == 0){
			context.unlockBuffer(this);
			
		}else{
			context.lockBuffer(this);
	
		}
	}
    ```
##  package org.wltea.analyzer.seg.LetterSegmenter

*   负责处理字母的子分词器，涵盖一下范围
    >1. 英文单词、英文加阿拉伯数字、专有名词（公司名）
    >2. IP地址、Email、URL

*   算法认为字母词语分成两类：混合字符组成、纯英文字母
    >1. 纯英文字母：a-z A-Z
    >2. 混合字符组成成分:
        >>>1.   纯英文字母
        >>>2.   阿拉伯数字 0-9
        >>>3.   连接字符: {'+','-','_','.','@','&','/','\\'}
*   具体算法略，和汉字切分流程差不多，从前往后连续地处理一个字符，上述两种情况分     别处理,
    
    ```
    public void nextLexeme(char[] segmentBuff , Context context) {

		//读取当前位置的char	
		char input = segmentBuff[context.getCursor()];
		
		boolean bufferLockFlag = false;
		//处理混合字母
		bufferLockFlag = this.processMixLetter(input, context) || bufferLockFlag;
		//处理英文字母
		bufferLockFlag = this.processEnglishLetter(input, context) || bufferLockFlag;
		//处理阿拉伯字母
        //bufferLockFlag = this.processPureArabic(input, context) || bufferLockFlag;
		
		//判断是否锁定缓冲区
		if(bufferLockFlag){
			context.lockBuffer(this);
		}else{
			//对缓冲区解锁
			context.unlockBuffer(this);
		}
	}
    ```
    
*   关键是枚举所有的英文字母、混合字母(阿拉伯)、字母之间可能存在的连接符
*   处理函数流程相似，==遇到非合理的字符，终止处理，认为已经完成一个词语的组装==
*   代码
    
    ```
    private boolean processMixLetter(char input , Context context){
		boolean needLock = false;
		
		if(start == -1){//当前的分词器尚未开始处理字符			
			if(isAcceptedCharStart(input)){
				//记录起始指针的位置,标明分词器进入处理状态
				start = context.getCursor();
				end = start;
			}
			
		}else{//当前的分词器正在处理字符			
			if(isAcceptedChar(input)){
				//输入不是连接符
				//if(!isLetterConnector(input)){
					//记录下可能的结束位置，如果是连接符结尾，则忽略
				//	end = context.getCursor();					
				//}
				//不在忽略尾部的链接字符
				end = context.getCursor();					
				
			}else{
				//生成已切分的词元
				Lexeme newLexeme = new Lexeme(context.getBuffOffset() , start , end - start + 1 , Lexeme.TYPE_LETTER);
				context.addLexeme(newLexeme);
				//设置当前分词器状态为“待处理”
				start = -1;
				end = -1;
			}			
		}
		
		//context.getCursor() == context.getAvailable() - 1读取缓冲区最后一个字符，直接输出
		if(context.getCursor() == context.getAvailable() - 1){
			if(start != -1 && end != -1){
				//生成已切分的词元
				Lexeme newLexeme = new Lexeme(context.getBuffOffset() , start , end - start + 1 , Lexeme.TYPE_LETTER);
				context.addLexeme(newLexeme);
			}
			//设置当前分词器状态为“待处理”
			start = -1;
			end = -1;
		}
		
		//判断是否锁定缓冲区
		if(start == -1 && end == -1){
			//对缓冲区解锁
			needLock = false;
		}else{
			needLock = true;
		}
		return needLock;
	}
    ```
##  package org.wltea.analyzer.seg.QuantifierSegmenter

*   类注释,数量词子分词器，涵盖一下范围
    >1. 阿拉伯数字，阿拉伯数字+中文量词===会分开输出
    >2. 中文数字+中文量词
    >3. 时间,日期
    >4. 罗马数字
    >5. 数学符号 % . / 

*   ==**数词的范围**==
    >1. 阿拉伯数字0-9
    >2. 阿拉伯数词链接符号  ",./:Ee"
    >3. 序数词（数词前缀）  "第初"
    >4. 中文数词    "○一二两三四五六七八九十零壹贰叁肆伍陆柒捌玖拾百千万亿拾佰仟萬億兆卅廿"
    >5. 中文数词连接符  "点"
    >6. 约数词（数词结尾） "几多余半"

*   数词状态变量定义
    ```
    //  阿拉伯数字0-9
	public static final int NC_ARABIC = 02;
	//  阿拉伯数字连接符
	public static final int NC_ANM = 03;
    // 序数词（说辞前缀）
    public static final int NC_NP = 11;
    //  中文数词
    public static final int NC_CHINESE = 12;
    //  中文数词连接符
    public static final int NC_CHINESE = 13;
    //  约数词（数词结尾）
    public static final int NC_NE = 14;
    //非数词字符
	public static final int NaN = -99;
    ```

*   类变量定义
    ```
    /*
	 * 词元的开始位置，
	 * 同时作为子分词器状态标识
	 * 当start > -1 时，标识当前的分词器正在处理字符
	 */
	private int nStart;
	/*
	 * 记录词元结束位置
	 * end记录的是在词元中最后一个出现的合理的数词结束
	 */
	private int nEnd;
	/*
	 * 当前数词的状态 
	 */
	private int nStatus;
	/*
	 * 捕获到一个数词
	 */
	private boolean fCaN;	
	
	/*
	 * 量词起始位置
	 */
	private int countStart;
	/*
	 * 量词终止位置
	 */
	private int countEnd;
    ```
*   主处理流程函数public void nextLexeme(char[] segmentBuff , Context context)
    ```
    public void nextLexeme(char[] segmentBuff , Context context) {
		fCaN = false;
		//数词处理部分
		processNumber(segmentBuff , context);
		
		//量词处理部分，同时判断是否存在量词		
		if(countStart == -1){//未开始处理量词
			//当前游标的位置紧挨着数词
			if((fCaN && nStart == -1)
					|| (nEnd != -1 && nEnd == context.getCursor() - 1)//遇到CNM的状态
					){				
				//量词处理
				processCount(segmentBuff , context);
			
			}
		}else{//已开始处理量词
			//量词处理
			processCount(segmentBuff , context);
		}

		//判断是否锁定缓冲区
		if(this.nStart == -1 && this.nEnd == -1 && NaN == this.nStatus
				&& this.countStart == -1 && this.countEnd == -1){
			//对缓冲区解锁
			context.unlockBuffer(this);
		}else{
			context.lockBuffer(this);
		}
	}
    ```
*   数词字符识别函数
    ```
	private int nIdentify(char[] segmentBuff , Context context){
		
		//读取当前位置的char	
		char input = segmentBuff[context.getCursor()];
		
		int type = NaN;
		//  非数词字符
		if(!AllNumberChars.contains(input)){
			return type;
		}
		
		if(CharacterHelper.isArabicNumber(input)){
			type = NC_ARABIC;
			 
		}else if(ChnNumberChars.contains(input)){
			type = NC_CHINESE;
			
		}else if(Num_Pre.indexOf(input) >= 0){
			type = NC_NP;
			
		}else if(Chn_Num_Mid.indexOf(input) >= 0){
			type = NC_CNM;
			
		}else if(NumEndChars.contains(input)){
			type = NC_NE;
			
		}else if(ArabicNumMidChars.contains(input)){
			type = NC_ANM;
		}
		return type;
	}
    ```
*   ==数词处理规则==
   
    ==主要判断之前处理的状态nStatus到当前字符状态inputStatus的转变是否         符合数字构成的规则==
        
*   数词处理函数private void processNumber(char[] segmentBuff , Context context)
   
    >1.函数源代码

    ```
    private void processNumber(char[] segmentBuff , Context context){		
		//数词字符识别,数词状态判断
		int inputStatus = nIdentify(segmentBuff , context);
		//  注意此处使用的是nStatus，并且一开始nStatus的值为NaN
		if(NaN == nStatus){
			//当前的分词器尚未开始处理字符
			onNaNStatus(inputStatus , context);
			
        //		}else if(NC_ANP == nStatus){ 
        //			//当前为阿拉伯数字前缀	
        //			onANPStatus(inputStatus , context);
			    
		}else if(NC_ARABIC == nStatus){
			//当前为阿拉伯数字
			onARABICStatus(inputStatus , context);
			
		}else if(NC_ANM	== nStatus){
			//当前为阿拉伯数字链接符
			onANMStatus(inputStatus , context);
			
        //		}else if(NC_ANE == nStatus){
        //			//当前为阿拉伯数字结束符
        //			onANEStatus(inputStatus , context);
			
		}else if(NC_NP == nStatus){
			//当前为中文数字前缀
			onNPStatus(inputStatus , context);
			
		}else if(NC_CHINESE == nStatus){
			//当前为中文数字
			onCHINESEStatus(inputStatus , context);
			
		}else if(NC_CNM == nStatus){
			//当前为中文数字连接符
			onCNMStatus(inputStatus , context);
			
		}else if(NC_NE == nStatus){
			//当前为中文数字结束符
			onCNEStatus(inputStatus , context);
			
        //		}else if(NC_ROME == nStatus){
        //			//当前为罗马数字
        //			onROMEStatus(inputStatus , context);			
			
		}
		
		//读到缓冲区最后一个字符，还有尚未输出的数词
		if(context.getCursor() == context.getAvailable() - 1){
			if(nStart != -1 && nEnd != -1){
				//输出数词
				outputNumLexeme(context);
			}
			//重置数词状态
			nReset();
		}				
	}
    ```
    >2.==**代码理解**==
        >>1.    nstatus是已经处理的数词的状态，一开始的初始状态为NaN,表明当前的分         词器尚未开始处理字符，因此一开始使用该量词分词器器的时候，会调用         数onNaNStatus，并且输入该函数的inputStatus的值不一定是NaN
        >>2.    ++数字输出的情况++
            >>>1.   遇到中文数词后缀，调用函数onCNEStatus
            >>>2.   判断当前输入字符的状态inputStatus和已经处理字符的状态nStatus         的一致性（当前输入的字符状态能否和之前的nstatus进行组合,组成         一个可能的数字，状态到状态的转变是合法的），如果不可以              ，重置或者输出之前的数字(通过调用函数outputNumLexeme)
            >>>3.   到缓冲区最后的时候

    >3.onCEStatus函数
    
    ```
    /**
	 * 当前为CNE状态时，状态机的处理(状态转换)
	 * @param inputStatus
	 * @param context
	 */
	private void onCNEStatus(int inputStatus ,  Context context){
		//输出可能存在的数词
		outputNumLexeme(context);
		//重置数词状态
		nReset();
		//进入初始态进行处理
		onNaNStatus(inputStatus , context);
				
	}
    ```
    >4.位置记录处理函数onNaNStatus 
    
    ```
        private void onNaNStatus(int inputStatus ,  Context context){
    		if(NaN == inputStatus){
    			return;
    			
    		}else if(NC_NP == inputStatus){//中文数词前缀
    			//记录起始位置
    			nStart = context.getCursor();
    			//记录当前的字符状态
    			nStatus = inputStatus;	
    			
    		}else if(NC_CHINESE == inputStatus){//中文数词
    			//记录起始位置
    			nStart = context.getCursor();
    			//记录当前的字符状态
    			nStatus = inputStatus;
    			//记录可能的结束位置
    			nEnd = context.getCursor();
    			
    		}else if(NC_NE == inputStatus){//中文数词后缀
    			//记录起始位置
    			nStart = context.getCursor();
    			//记录当前的字符状态
    			nStatus = inputStatus;
    			//记录可能的结束位置
    			nEnd = context.getCursor();
    			
    		//}else if(NC_ANP == inputStatus){//阿拉伯数字前缀
    			//记录起始位置
    			//nStart = context.getCursor();
    			//记录当前的字符状态
    			//nStatus = inputStatus;
    			
    		}else if(NC_ARABIC == inputStatus){//阿拉伯数字
    			//记录起始位置
    			nStart = context.getCursor();
    			//记录当前的字符状态
    			nStatus = inputStatus;
    			//记录可能的结束位置
    			nEnd = context.getCursor();
    			
    		//}else if(NC_ROME == inputStatus){//罗马数字
    			//记录起始位置
    			//nStart = context.getCursor();
    			//记录当前的字符状态
    			//nStatus = inputStatus;
    			//记录可能的结束位置
    			//nEnd = context.getCursor();	
    		
    		}else{
    			//对NC_ANM ，NC_ANE和NC_CNM 不做处理
    		}
    	}
    ```
*   一致性判断函数
    ```
    /**
	 *  当前为NP状态时，状态机的处理(状态转换)
	 * @param inputStatus
	 * @param context
	 */
	private void onNPStatus(int inputStatus ,  Context context){
		if(NC_CHINESE == inputStatus){//中文数字
			//记录可能的结束位置
			nEnd = context.getCursor();
			//记录当前的字符状态
			nStatus = inputStatus;

			
		}else if(NC_ARABIC == inputStatus){//阿拉伯数字
			//记录可能的结束位置
			nEnd = context.getCursor();
			//记录当前的字符状态
			nStatus = inputStatus;
			
        //}else if(NC_ROME == inputStatus){//罗马数字
            //记录可能的结束位置
            //nEnd = context.getCursor() - 1;
		    //输出可能存在的数词
            //outputNumLexeme(context);
			//重置数词状态
            //nReset();
			//进入初始态进行处理
            //onNaNStatus(inputStatus , context);	
			
		}else{
			//重置数词状态,没有输出
			nReset();
			//进入初始态进行处理
			onNaNStatus(inputStatus , context);
			
		}
	}
    ```
    
    ```
    private void onANMStatus(int inputStatus ,  Context context){
		if (NC_ARABIC == inputStatus){//阿拉伯数字
			//记录当前的字符状态
			nStatus = inputStatus;
			//记录可能的结束位置
			nEnd = context.getCursor();
			
		//}else if (NC_ANP == inputStatus){//阿拉伯数字前缀
			//记录当前的字符状态
			//nStatus = inputStatus;
			
		}else{
			//输出可能存在的数词
			outputNumLexeme(context);
			//重置数词状态
			nReset();
			//进入初始态进行处理
			onNaNStatus(inputStatus , context);
			
		}		
	}
    ```
*  量词，通过两次匹配词典的方式进行匹配
    ```
    private void processCount(char[] segmentBuff , Context context){
		Hit hit = null;

		if(countStart == -1){
			hit = Dictionary.matchInQuantifierDict(segmentBuff , context.getCursor() , 1);
		}else{
			hit = Dictionary.matchInQuantifierDict(segmentBuff , countStart , context.getCursor() - countStart + 1);
		}
		
		if(hit != null){
			if(hit.isPrefix()){
				if(countStart == -1){
					//设置量词的开始
					countStart = context.getCursor();
				}
			}
			
			if(hit.isMatch()){
				if(countStart == -1){
					countStart = context.getCursor();
				}
				//设置量词可能的结束
				countEnd = context.getCursor();
				//输出可能存在的量词
				outputCountLexeme(context);
			}
			
			if(hit.isUnmatch()){
				if(countStart != -1){
					//重置量词状态
					countStart = -1;
					countEnd = -1;
				}
			}
		}
		
		//读到缓冲区最后一个字符，还有尚未输出的量词
		if(context.getCursor() == context.getAvailable() - 1){
			//重置量词状态
			countStart = -1;
			countEnd = -1;
		}
	}
    ```